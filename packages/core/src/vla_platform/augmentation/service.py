import asyncio
import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path

import httpx

from vla_platform.api_connections import ApiConnections, ApiKeyError
from vla_platform.augmentation.contracts import (
    AugmentationClip,
    AugmentationOptions,
    AugmentationPlan,
    AugmentationRequest,
    AugmentationResult,
)
from vla_platform.augmentation.media import (
    MAX_INPUT_BYTES,
    MAX_OUTPUT_BYTES,
    duration,
    trim,
    verify_output,
)
from vla_platform.augmentation.provider import CLOUD_MODEL, MODEL, GeminiOmniProvider
from vla_platform.augmentation.source import download_video
from vla_platform.cloud_connections import CloudConnections, GcpConnectionConfig
from vla_platform.contracts import TERMINAL
from vla_platform.datasets.explore import DatasetExplorer, source_profile
from vla_platform.modules import read_registry


class Augmentation:
    def __init__(self, execution):
        self.execution = execution
        self.slots = asyncio.Semaphore(1)

    def integration_installed(self, ident: str) -> bool:
        try:
            registry = read_registry(self.execution.settings.data_dir)
            return registry is None or ident in registry.installed
        except ValueError:
            return False

    def api_key(self) -> str | None:
        if not self.integration_installed("gemini"):
            return None
        try:
            return ApiConnections(self.execution.settings.data_dir).key("gemini")
        except ApiKeyError, OSError:
            return None

    def options(self) -> AugmentationOptions:
        missing = []
        # Re-read the saved non-secret configuration so connect/disconnect in
        # Settings takes effect immediately for both web and CLI callers.
        connections = CloudConnections(self.execution.settings.data_dir).list()
        cloud = next((item for item in connections.providers if item.provider == "gcp"), None)
        config = cloud.config if cloud else None
        project = (
            config.project_id
            if isinstance(config, GcpConnectionConfig)
            and self.integration_installed("google-cloud")
            else None
        )
        auth_mode = "unconfigured"
        auth_message = "Connect Google Cloud in Cloud & APIs to use Gemini Omni without an API key."
        if project:
            auth_mode = "google_cloud"
            auth_message = (
                f"Google Cloud · {project} · uses your Google Cloud login, without an API key. "
                "Model access and billing are checked when a run starts."
            )
            if not shutil.which("gcloud"):
                missing.append(
                    "Google Cloud CLI on PATH and reconnect Google Cloud in Cloud & APIs"
                )
        elif self.api_key():
            auth_mode = "gemini_api_key"
            auth_message = (
                "Gemini Omni API key configured. "
                "Connect Google Cloud in Cloud & APIs to use your login."
            )
        else:
            if not self.integration_installed("google-cloud") and not self.integration_installed(
                "gemini"
            ):
                missing.append("a Google Cloud or Gemini Omni integration from Modules")
            elif self.integration_installed("gemini"):
                missing.append("a Gemini Omni key in Cloud & APIs")
            else:
                missing.append("Google Cloud in Cloud & APIs")
        for binary in ("ffmpeg", "ffprobe"):
            if not shutil.which(binary):
                missing.append(binary + " on PATH")
        return AugmentationOptions(
            configured=not missing,
            model=CLOUD_MODEL if project else MODEL,
            auth_mode=auth_mode,
            google_cloud_project=project,
            auth_message=auth_message,
            setup_message="Configure " + ", ".join(missing) + "." if missing else None,
        )

    async def validate(self, project_id: str, request: AugmentationRequest):
        options = self.options()
        if not options.configured:
            raise ValueError(options.setup_message)
        source = await self.execution.get(request.source_job_id)
        if source is None or source.project_id != project_id:
            raise ValueError("Choose an inspected dataset in this project")
        profile = source_profile(source)
        if any(index >= profile.total_episodes for index in request.episode_indices):
            raise ValueError("An episode index is outside the inspected dataset")
        feature = profile.features.get(request.camera_key)
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            raise ValueError("Choose a video camera from the inspected dataset")
        if request.dataset_scope:
            for key in request.dataset_scope.camera_keys:
                if profile.features.get(key, {}).get("dtype") != "video":
                    raise ValueError("Choose video cameras from the inspected dataset")
            await self.recipe(project_id, request)
        if request.source_clip:
            await self.saved_clip(project_id, request)
        return source

    async def recipe(self, project_id, request):
        """Resolve the reviewed edit chain, including refinements, from saved records."""
        current = await self.execution.get(request.dataset_scope.recipe_job_id)
        selected, prompts, visited, include_output = current, [], set(), True
        while current is not None:
            if (
                current.id in visited
                or len(visited) >= 16
                or current.project_id != project_id
                or current.status != "succeeded"
                or not isinstance(current.request, AugmentationRequest)
                or not isinstance(current.result, AugmentationResult)
                or current.request.source_job_id != request.source_job_id
                or current.request.dataset_scope
            ):
                raise ValueError("Choose a completed preview recipe from this dataset and project")
            visited.add(current.id)
            if include_output:
                prompts.insert(0, current.result.prompt)
            parent = current.request.source_clip
            if not parent:
                break
            include_output = parent.variant == "augmented"
            current = await self.execution.get(parent.job_id)
            if current is None:
                raise ValueError("A saved recipe input is missing")
        if selected is None or not prompts:
            raise ValueError("Choose a completed preview recipe")
        if request.prompt != selected.request.prompt or request.preset != selected.request.preset:
            raise ValueError("The dataset recipe must match the reviewed preview")
        return prompts

    async def dataset_plan(self, project_id, request, explorer=None):
        source = await self.validate(project_id, request)
        if not request.dataset_scope:
            raise ValueError("Choose a full-dataset recipe and cameras")
        profile = source_profile(source)
        if not 0 < profile.total_episodes <= 1000:
            raise ValueError("Full-dataset augmentation supports 1–1,000 episodes per run")
        prompts = await self.recipe(project_id, request)
        own_explorer = explorer is None
        explorer = explorer or DatasetExplorer(data_dir=self.execution.settings.data_dir)
        segments = []
        try:
            # Metadata only: no provider construction or camera-media downloads.
            for episode in range(profile.total_episodes):
                preview = await explorer.preview(source, episode)
                for key in request.dataset_scope.camera_keys:
                    camera = next((c for c in preview.cameras if c.key == key), None)
                    if (
                        camera is None
                        or camera.end_seconds - camera.start_seconds + 0.001
                        < preview.duration_seconds
                    ):
                        raise ValueError(f"Episode {episode} does not have full coverage for {key}")
                    seconds = preview.duration_seconds
                    if not math.isfinite(seconds) or seconds < 1:
                        raise ValueError(f"Episode {episode} must contain at least one second")
                    count = math.ceil(seconds / 10)
                    for part in range(count):
                        start = camera.start_seconds + seconds * part / count
                        end = camera.start_seconds + seconds * (part + 1) / count
                        segments.append((episode, key, start, end, camera.url))
                    if len(segments) * len(prompts) > 1000:
                        raise ValueError(
                            "This plan exceeds 1,000 generation requests; "
                            "choose fewer cameras or a smaller dataset"
                        )
        finally:
            if own_explorer:
                await explorer.close()
        options = self.options()
        material = dict(
            source=source.id,
            revision=profile.revision,
            metadata=profile.metadata_sha256,
            segments=segments,
            prompts=prompts,
            model=options.model,
            auth_mode=options.auth_mode,
            cloud_project=options.google_cloud_project,
        )
        digest = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
        plan = AugmentationPlan(
            total_episodes=profile.total_episodes,
            camera_keys=request.dataset_scope.camera_keys,
            total_source_seconds=sum(end - start for _, _, start, end, _ in segments),
            total_clips=len(segments),
            provider_requests=len(segments) * len(prompts),
            recipe_prompts=prompts,
            plan_sha256=digest,
            model=options.model,
            warnings=[
                "Full episodes are covered in segments of at most 10 seconds. "
                "Appearance may vary at segment boundaries.",
                "Cameras are generated independently; "
                "visual consistency between viewpoints is not guaranteed.",
                "Outputs are review clips, not an automatically registered training dataset. "
                "Original data and actions stay unchanged.",
            ],
        )
        return plan, segments

    async def prepare_dataset(self, job, request, directory, explorer):
        await self.stage(job.id, "Checking full episodes and selected cameras")
        plan, segments = await self.dataset_plan(job.project_id, request, explorer)
        if request.dataset_scope.plan_sha256 != plan.plan_sha256:
            raise ValueError("The dataset plan changed. Review it again before generating")
        required = len(segments) * (MAX_INPUT_BYTES + MAX_OUTPUT_BYTES) * 2 + 11 * 1024**3
        if shutil.disk_usage(directory).free < required:
            raise ValueError("Not enough disk space for this dataset plan and review bundle")
        (directory / "plan.json").write_text(plan.model_dump_json(indent=2))
        grouped = {}
        for index, (episode, key, start, end, url) in enumerate(segments):
            grouped.setdefault(url, []).append((index, episode, key, start, end))
        prepared = []
        for url, items in grouped.items():
            await self.stage(
                job.id, f"Preparing full-dataset clips ({len(prepared)}/{len(segments)})"
            )
            video = directory / "source.mp4"
            try:
                await download_video(explorer.client, url, video)
                for index, episode, key, start, end in items:
                    await self.stage(
                        job.id,
                        f"Preparing full-dataset clip {len(prepared) + 1} of {len(segments)}",
                    )
                    original = directory / f"original-{index}.mp4"
                    await trim(video, original, start, end - start)
                    prepared.append((index, episode, start, end, original, key))
            finally:
                video.unlink(missing_ok=True)
        return sorted(prepared), plan

    async def saved_clip(self, project_id: str, request: AugmentationRequest):
        reference = request.source_clip
        parent = await self.execution.get(reference.job_id)
        if (
            parent is None
            or parent.project_id != project_id
            or parent.status != "succeeded"
            or not isinstance(parent.request, AugmentationRequest)
            or not isinstance(parent.result, AugmentationResult)
        ):
            raise ValueError("Choose a completed augmentation in this project")
        clip = next(
            (item for item in parent.result.clips if item.index == reference.clip_index), None
        )
        if clip is None:
            raise ValueError("The selected augmentation clip does not exist")
        if (
            parent.result.source_job_id != request.source_job_id
            or parent.request.source_job_id != request.source_job_id
            or request.episode_indices != [clip.episode_index]
            or request.camera_key != clip.camera_key
            or request.start_seconds != parent.request.start_seconds
            or abs(request.duration_seconds - (clip.source_end_seconds - clip.source_start_seconds))
            > 0.001
        ):
            raise ValueError(
                "Saved clip selection must keep its dataset, episode, camera and timing"
            )
        path = await self.download(parent.id, clip.index, original=reference.variant == "original")
        if path.is_symlink() or not 0 < path.stat().st_size <= MAX_OUTPUT_BYTES:
            raise ValueError("The saved source clip is invalid or exceeds the 40 MiB limit")
        return clip, path

    async def prepare_saved_clip(self, project_id, request, directory):
        clip, path = await self.saved_clip(project_id, request)
        video = directory / "source.mp4"
        original = directory / "original-0.mp4"
        try:
            shutil.copyfile(path, video)
            if video.stat().st_size > MAX_OUTPUT_BYTES:
                raise ValueError("The saved source clip exceeds the 40 MiB limit")
            with video.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            expected = (
                clip.input_sha256
                if request.source_clip.variant == "original"
                else clip.output_sha256
            )
            if digest != expected:
                raise ValueError("The saved source clip no longer matches its recorded hash")
            actual = await duration(video)
            if abs(actual - request.duration_seconds) > 0.15:
                raise ValueError("The saved source clip no longer matches its recorded duration")
            if video.stat().st_size >= MAX_INPUT_BYTES:
                await trim(video, original, 0, request.duration_seconds)
            else:
                # Preserve the exact reviewed input bytes whenever they fit the upload limit.
                video.replace(original)
        finally:
            video.unlink(missing_ok=True)
        return [
            (0, clip.episode_index, clip.source_start_seconds, clip.source_end_seconds, original)
        ]

    async def stage(self, job_id: str, text: str):
        async with self.execution.lock:
            job = await self.execution.get(job_id)
            if job is None or job.status in TERMINAL:
                # A transport may return a late response after cancellation. Never
                # let that response trigger the next paid request in a batch.
                raise asyncio.CancelledError
            job.stage = text
            await self.execution.save(job)

    async def run(self, job):
        explorer = None
        provider = None
        try:
            async with self.slots:
                async with self.execution.lock:
                    job = await self.execution.get(job.id)
                    if job.status in TERMINAL:
                        return
                    job.status = "running"
                    await self.execution.save(job)
                async with asyncio.timeout(86400 if job.request.dataset_scope else 1800):
                    request = job.request
                    source = await self.validate(job.project_id, request)
                    profile = source_profile(source)
                    directory = self.execution.settings.data_dir / "jobs" / job.id
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / "request.json").write_text(request.model_dump_json(indent=2))
                    options = self.options()
                    provider = (
                        GeminiOmniProvider(google_cloud_project=options.google_cloud_project)
                        if options.auth_mode == "google_cloud"
                        else GeminiOmniProvider(self.api_key())
                    )
                    plan = None
                    if request.dataset_scope:
                        explorer = DatasetExplorer(data_dir=self.execution.settings.data_dir)
                        prepared, plan = await self.prepare_dataset(
                            job, request, directory, explorer
                        )
                    elif request.source_clip:
                        await self.stage(job.id, "Preparing saved clip")
                        prepared = await self.prepare_saved_clip(job.project_id, request, directory)
                    else:
                        explorer = DatasetExplorer(data_dir=self.execution.settings.data_dir)
                        prepared = []
                        # Validate and trim the entire selection before any paid generation request.
                        for index, episode in enumerate(request.episode_indices):
                            await self.stage(job.id, f"Preparing clip {index + 1}")
                            preview = await explorer.preview(source, episode)
                            camera = next(
                                (
                                    item
                                    for item in preview.cameras
                                    if item.key == request.camera_key
                                ),
                                None,
                            )
                            if camera is None:
                                raise ValueError(
                                    "The selected camera has no playable episode video"
                                )
                            start = camera.start_seconds + request.start_seconds
                            end = start + request.duration_seconds
                            if (
                                end
                                > min(
                                    camera.end_seconds,
                                    camera.start_seconds + preview.duration_seconds,
                                )
                                + 0.001
                            ):
                                raise ValueError(
                                    f"Episode {episode} is too short "
                                    "for the requested clip interval"
                                )
                            video = directory / "source.mp4"
                            original = directory / f"original-{index}.mp4"
                            await download_video(explorer.client, camera.url, video)
                            try:
                                await trim(video, original, start, request.duration_seconds)
                            finally:
                                video.unlink(missing_ok=True)
                            prepared.append((index, episode, start, end, original))
                    if not request.dataset_scope:
                        prepared = [(*item, request.camera_key) for item in prepared]
                    prompts = plan.recipe_prompts if plan else [request.resolved_prompt()]
                    prompt = "\n\nThen apply:\n".join(prompts)
                    clips = []
                    for index, episode, start, end, original, camera_key in prepared:
                        await self.stage(job.id, f"Augmenting clip {index + 1} of {len(prepared)}")
                        input_bytes = original.read_bytes()
                        output = directory / f"augmented-{index}.mp4"
                        working = input_bytes
                        for step, edit in enumerate(prompts):
                            await self.stage(
                                job.id,
                                f"Augmenting clip {index + 1} of {len(prepared)} · "
                                f"edit {step + 1} of {len(prompts)}",
                            )
                            output_bytes, interaction_id = await provider.edit(working, edit)
                            output.write_bytes(output_bytes)
                            await verify_output(output, end - start)
                            if step < len(prompts) - 1 and len(output_bytes) >= MAX_INPUT_BYTES:
                                intermediate = directory / "intermediate.mp4"
                                try:
                                    await trim(output, intermediate, 0, end - start)
                                    working = intermediate.read_bytes()
                                finally:
                                    intermediate.unlink(missing_ok=True)
                            else:
                                working = output_bytes
                        clips.append(
                            AugmentationClip(
                                index=index,
                                episode_index=episode,
                                camera_key=camera_key,
                                source_start_seconds=start,
                                source_end_seconds=end,
                                input_sha256=hashlib.sha256(input_bytes).hexdigest(),
                                output_sha256=hashlib.sha256(output_bytes).hexdigest(),
                                interaction_id=interaction_id,
                            )
                        )
                    result = AugmentationResult(
                        model=options.model,
                        auth_mode=options.auth_mode,
                        google_cloud_project=options.google_cloud_project,
                        prompt=prompt,
                        source_job_id=source.id,
                        source_clip=request.source_clip,
                        dataset_scope=request.dataset_scope,
                        recipe_prompts=prompts if plan else None,
                        repo_id=profile.repo_id,
                        revision=profile.revision,
                        metadata_sha256=profile.metadata_sha256,
                        clips=clips,
                        warnings=[
                            "Generated appearance may change geometry, motion or contact. "
                            "Review each clip before pairing it with original actions.",
                            "Clip timing is checked, but frame alignment and action validity "
                            "are not verified. Outputs are not registered as training data.",
                            "The source dataset and its action/state records are unchanged.",
                            *(
                                plan.warnings
                                if plan
                                else ["Only the selected camera and time intervals are augmented."]
                            ),
                        ],
                    )
                    manifest = directory / "manifest.json"
                    manifest.write_text(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "request": request.model_dump(),
                                "result": result.model_dump(),
                                "files": [
                                    {
                                        "index": clip.index,
                                        "original": f"original-{clip.index}.mp4",
                                        "augmented": f"augmented-{clip.index}.mp4",
                                    }
                                    for clip in clips
                                ],
                            },
                            indent=2,
                        )
                    )
                    # Video is already compressed; ZIP_STORED keeps packaging bounded and quick.
                    with zipfile.ZipFile(directory / "augmentation.zip", "w") as archive:
                        archive.write(manifest, manifest.name)
                        if plan:
                            archive.write(directory / "plan.json", "plan.json")
                        for clip in clips:
                            for prefix in ("original", "augmented"):
                                path = directory / f"{prefix}-{clip.index}.mp4"
                                archive.write(path, path.name)
                    async with self.execution.lock:
                        current = await self.execution.get(job.id)
                        if current.status not in TERMINAL:
                            current.result = result
                            current.status = "succeeded"
                            current.stage = "Ready for review"
                            await self.execution.save(current)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if isinstance(exc, TimeoutError):
                message = "Augmentation timed out. No request was retried automatically."
            elif isinstance(exc, httpx.HTTPError):
                message = "Could not fetch dataset media. Check connectivity and retry explicitly."
            elif isinstance(exc, ValueError):
                message = str(exc)[:1000]
            else:
                message = "Augmentation failed while processing media. No output was published."
            async with self.execution.lock:
                current = await self.execution.get(job.id)
                if current.status not in TERMINAL:
                    current.status, current.error = "failed", message
                    await self.execution.save(current)
        finally:
            if provider:
                await provider.close()
            if explorer:
                await explorer.close()

    async def download(self, job_id: str, index: int | None = None, original=False) -> Path:
        job = await self.execution.get(job_id)
        if not job or job.status != "succeeded" or not isinstance(job.result, AugmentationResult):
            raise ValueError("Finish a successful augmentation before downloading its outputs")
        if index is not None and index not in [clip.index for clip in job.result.clips]:
            raise ValueError("Augmentation clip not found")
        filename = (
            "augmentation.zip"
            if index is None
            else (f"{'original' if original else 'augmented'}-{index}.mp4")
        )
        path = self.execution.settings.data_dir / "jobs" / job.id / filename
        if not path.is_file():
            raise ValueError("The augmentation output file is missing")
        return path
