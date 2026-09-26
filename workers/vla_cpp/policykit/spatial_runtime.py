"""Shared offline native/C++ SmolVLA observation-to-action implementation.

Application callers supply verified local package paths. This module never
resolves Hub IDs or downloads weights. Import only inside an isolated ML worker.
"""

from __future__ import annotations

import contextlib
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

LM_ROOT = "model.vlm_with_expert.vlm.model.text_model.layers."


def load_smolvla_config(path):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    # Dispatch on the serialized "type" before decoding the concrete dataclass.
    config = PreTrainedConfig.from_pretrained(path)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError("Expected a SmolVLA checkpoint configuration")
    return config


def load_checkpoint_weights(policy, model_file):
    from safetensors.torch import load_file

    # The constructor uses mixed dtypes that can differ from released weights.
    # Preserve checkpoint dtypes while still requiring every key and shape.
    policy.load_state_dict(load_file(str(model_file), device="cpu"), strict=True, assign=True)
    return policy


def quantize_linears(policy, roots):
    """Load FP weights on CPU, then pack selected layers directly onto CUDA one at a time."""
    import bitsandbytes as bnb
    import torch

    names = []
    for name, layer in list(policy.named_modules()):
        if not isinstance(layer, torch.nn.Linear) or not name.startswith(roots):
            continue
        if isinstance(layer, bnb.nn.Linear4bit):
            raise ValueError(f"Already quantized: {name}")
        packed = bnb.nn.Linear4bit(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            compute_dtype=torch.bfloat16,
            compress_statistics=True,
            quant_type="nf4",
            # SmolVLA explicitly casts activations to q/k/v/o_proj.weight.dtype.
            # Floating *storage* preserves those casts; values are still packed 4-bit NF4.
            quant_storage=torch.bfloat16,
        )
        packed.load_state_dict(layer.state_dict())
        packed.requires_grad_(False)
        packed = packed.to("cuda:0")
        parent_name, _, child_name = name.rpartition(".")
        policy.get_submodule(parent_name).set_submodule(child_name, packed)
        names.append(name)
    if not names:
        raise ValueError(
            "No supported SmolVLA linear layers found; refusing an unquantized candidate"
        )
    policy.is_loaded_in_4bit = True  # PEFT dispatches to its bitsandbytes LoRA layer.
    return names


def patch_int8_activation_casts():
    """Patch only activation dtype lookups, in memory, for LeRobot 0.4.4.

    Packed INT8 storage must never become an activation dtype. Floating layers
    keep exactly the upstream dtype; the floating-reference probe checks parity.
    """
    import ast
    import inspect
    import textwrap

    import bitsandbytes as bnb
    import torch
    from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel

    def activation_dtype(layer):
        if layer.weight.is_floating_point():
            return layer.weight.dtype
        if isinstance(layer, bnb.nn.Linear8bitLt):
            return torch.bfloat16
        raise TypeError(f"Unsupported integer weight storage in {type(layer)}")

    class Rewrite(ast.NodeTransformer):
        count = 0

        def visit_Attribute(self, node):
            if (
                node.attr == "dtype"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "weight"
            ):
                self.count += 1
                return ast.copy_location(
                    ast.Call(
                        func=ast.Name(id="_activation_dtype", ctx=ast.Load()),
                        args=[node.value.value],
                        keywords=[],
                    ),
                    node,
                )
            return self.generic_visit(node)

    patched = {}
    for name in ("forward_attn_layer", "forward_cross_attn_layer", "forward"):
        original = getattr(SmolVLMWithExpertModel, name)
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
        rewrite = Rewrite()
        tree = ast.fix_missing_locations(rewrite.visit(tree))
        namespace = dict(original.__globals__, _activation_dtype=activation_dtype)
        exec(compile(tree, f"<smolvla-int8-{name}>", "exec"), namespace)
        patched[name] = (namespace[name], rewrite.count)
    if sum(count for _, count in patched.values()) != 7:
        raise RuntimeError("LeRobot activation cast layout changed; refusing unchecked patch")
    for name, (method, _) in patched.items():
        setattr(SmolVLMWithExpertModel, name, method)
    return {name: count for name, (_, count) in patched.items()}


def noise(seed):
    return torch.from_numpy(
        np.random.default_rng(seed).standard_normal((1, 50, 32)).astype("float32")
    )


class Runtime:
    def __init__(
        self,
        backend,
        root,
        output,
        *,
        policy_path,
        backbone_path,
        model_path=None,
        build=None,
        vendor=None,
    ):
        self.server = self.sock = self.log = self.proto_dir = None
        try:
            self._initialize(
                backend,
                root,
                output,
                policy_path=policy_path,
                backbone_path=backbone_path,
                model_path=model_path,
                build=build,
                vendor=vendor,
            )
        except Exception:
            self.close()
            raise

    def _initialize(
        self, backend, root, output, *, policy_path, backbone_path, model_path, build, vendor
    ):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        started = time.perf_counter()
        self.backend, self.server, self.sock = backend, None, None
        self.root, self.output = root, output
        self.path = str(Path(policy_path).resolve())
        backbone = str(Path(backbone_path).resolve())
        self.config = load_smolvla_config(self.path)
        self.config.pretrained_path = Path(self.path)
        self.config.device = "cpu"
        self.config.vlm_model_name = backbone
        self.config.load_vlm_weights = False
        self.config.compile_model = False
        self.pre, self.post = make_pre_post_processors(
            self.config,
            pretrained_path=self.path,
            preprocessor_overrides={
                "device_processor": {"device": "cpu"},
                "tokenizer_processor": {"tokenizer_name": backbone},
            },
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.prepare_images = lambda batch: SmolVLAPolicy.prepare_images(self, batch)[0]
        if backend.startswith("native-"):
            precision = backend.removeprefix("native-")
            self.dtype = {"f32": torch.float32, "fp16": torch.float16}.get(
                precision, torch.bfloat16
            )
            self.policy = load_checkpoint_weights(
                SmolVLAPolicy(self.config), Path(self.path) / "model.safetensors"
            )
            self.policy.requires_grad_(False).eval().to(dtype=self.dtype)
            if precision == "nf4":
                quantize_linears(self.policy, roots=(LM_ROOT,))
            elif precision == "int8":
                import bitsandbytes as bnb

                for name, layer in list(self.policy.named_modules()):
                    if name.startswith(LM_ROOT) and isinstance(layer, torch.nn.Linear):
                        packed = bnb.nn.Linear8bitLt(
                            layer.in_features,
                            layer.out_features,
                            bias=layer.bias is not None,
                            has_fp16_weights=False,
                            threshold=6.0,
                        )
                        packed.load_state_dict(layer.state_dict())
                        packed.requires_grad_(False).to("cuda")
                        parent, _, child = name.rpartition(".")
                        self.policy.get_submodule(parent).set_submodule(child, packed)
            if precision == "int8":
                patch_int8_activation_casts()
            self.policy.to("cuda")
            self.config.device = "cuda"
        else:
            import importlib
            import tempfile

            from grpc_tools import protoc

            self.proto_dir = tempfile.TemporaryDirectory(prefix="smolvla-proto-")
            proto = (Path(vendor) if vendor else root / "vla.cpp") / "src/serving/vla.proto"
            code = protoc.main(
                ["protoc", f"-I{proto.parent}", f"--python_out={self.proto_dir.name}", str(proto)]
            )
            if code:
                raise RuntimeError("Protobuf generation failed")
            sys.path.insert(0, self.proto_dir.name)
            self.pb = importlib.import_module("vla_pb2")
            import zmq

            with socket.socket() as port:
                port.bind(("127.0.0.1", 0))
                number = port.getsockname()[1]
            address = f"tcp://127.0.0.1:{number}"
            self.log = (output / "server.log").open("w")
            self.server = subprocess.Popen(
                [
                    str((Path(build) if build else root / "vla.cpp/build-cuda") / "vla-server"),
                    "--bind",
                    address,
                    str(
                        model_path or root / "gguf" / f"smolvla-{backend.removeprefix('cpp-')}.gguf"
                    ),
                ],
                stdout=self.log,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 90
            while True:
                if self.server.poll() is not None:
                    raise RuntimeError("C++ server exited")
                try:
                    with socket.create_connection(("127.0.0.1", number), timeout=0.2):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("C++ server startup")
                    time.sleep(0.1)
            self.sock = zmq.Context.instance().socket(zmq.REQ)
            self.sock.setsockopt(zmq.RCVTIMEO, 120000)
            self.sock.setsockopt(zmq.LINGER, 0)
            self.sock.connect(address)
            server_text = (output / "server.log").read_text()
            if "backend = CUDA" not in server_text or "falling back to CPU" in server_text:
                raise RuntimeError("C++ CUDA backend not established")
        self.startup_s = time.perf_counter() - started
        self.startup_peak_bytes = (
            torch.cuda.max_memory_allocated() if backend.startswith("native-") else None
        )

    def predict(self, raw, initial_noise):
        if (
            initial_noise.shape != (1, 50, 32)
            or initial_noise.dtype != torch.float32
            or initial_noise.device.type != "cpu"
            or not torch.isfinite(initial_noise).all()
        ):
            raise ValueError("Noise must be finite CPU FP32 [1,50,32]")
        tick = time.perf_counter()
        batch = self.pre(dict(raw))
        if self.backend.startswith("native-"):
            batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in batch.items()}
            self.policy.reset()
            amp = (
                torch.autocast("cuda", dtype=self.dtype)
                if self.dtype != torch.float32
                else contextlib.nullcontext()
            )
            with torch.inference_mode(), amp:
                actions = (
                    self.post(
                        self.policy.predict_action_chunk(batch, noise=initial_noise.to("cuda"))
                    )
                    .float()
                    .cpu()
                )
            torch.cuda.synchronize()
        else:
            req = self.pb.PredictRequest()
            for img in self.prepare_images(batch):
                # Native image resizing and [-1,1] transform are shared. Server
                # applies the inverse affine range transform exactly once.
                arr = ((img[0] + 1) / 2).permute(1, 2, 0).contiguous().numpy()
                field = req.images.add()
                field.encoding = self.pb.Image.F32_RGB_01
                field.height, field.width = arr.shape[:2]
                field.data = arr.tobytes()
            mask = batch["observation.language.attention_mask"][0].bool()
            req.lang_tokens.extend(batch["observation.language.tokens"][0][mask].tolist())
            # C++ owns state normalization and action unnormalization, using GGUF stats.
            state = raw["observation.state"].reshape(-1).tolist()
            req.state.extend(state + [0.0] * (32 - len(state)))
            req.noise.extend(initial_noise.reshape(-1).tolist())
            self.sock.send(req.SerializeToString())
            response = self.pb.PredictResponse()
            response.ParseFromString(self.sock.recv())
            if response.error:
                raise RuntimeError(response.error)
            actions = torch.tensor(response.action_chunk).reshape(
                1, response.chunk_size, response.action_dim
            )[:, :, :7]
        if actions.shape != (1, 50, 7) or not torch.isfinite(actions).all():
            raise ValueError("Invalid action chunk")
        return actions, 1000 * (time.perf_counter() - tick)

    def close(self):
        if self.sock is not None:
            self.sock.close(linger=0)
        if self.server is not None:
            self.server.terminate()
            try:
                self.server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait()
        if self.log is not None:
            self.log.close()
        if self.proto_dir is not None:
            self.proto_dir.cleanup()


def save_fixture(path, raw, initial_noise):
    values = {k: v.cpu().numpy() for k, v in raw.items() if torch.is_tensor(v)}
    values["task"] = np.array(raw["task"])
    values["noise"] = initial_noise.numpy()
    np.savez_compressed(path, **values)


def load_fixture(path):
    from .spatial_fixture import load_arrays

    values = load_arrays(path)
    raw = {k: torch.from_numpy(v) for k, v in values.items() if k not in ("task", "noise")}
    raw["task"] = values["task"].tolist()
    return raw, torch.from_numpy(values["noise"])
