# Local ACT distillation evidence

`local-cpu.json` binds the implemented worker, exact runtime source,41 executor
tests and a separate experiment with the user's unchanged supplied ACT teacher.
Four genuine CPU updates produced55,971,416-byte student weights. The comparison
baseline is136,972,568 bytes of teacher inference tensors, excluding its VAE;
the original206,699,736-byte file is also recorded without counting VAE removal
as a distillation benefit. These quantities exclude package metadata.

Six generated observations were partitioned into three explicit generated groups.
Validation normalized imitation L1 changed from0.6127 to0.3386; final imitation
L1 was0.3903 after freezing the selected last update. This is a small algorithm
check, not a robot generalization benchmark or evidence of task performance.
Recorded demonstration diagnostics are per-coordinate to avoid silently combining
arm and gripper units. The observed11.36seconds is execution time after initial
admission on this machine, not an inference speedup or cross-hardware benchmark.

Fresh-process reload reproduced every full100x6 raw/postprocessed action chunk
and action queue/reset; all9 published payload file hashes were checked. The
native model identifier matches the existing simulation checkpoint inspector.
The actual teacher and corpus were unchanged. No model/recording bytes are in Git.

A separate native-format bridge proof used the genuine LeRobot writer, existing
snapshot/media validation and native reader on24 generated frames,6 episodes and
3 scene groups. It produced12 bounded observations. Its initial output policy
predated the later native config-type/model-ID correction; it is **not** reused as
final package acceptance evidence. The final41 suite and supplied-teacher artifact
establish that correction. The preparation sampling optimization has source review
but needs a final native-reader rerun before application integration acceptance.

No new dependencies, downloads, GPU/provider/cloud operations, or user credential
changes occurred. There is no task-success, calibration, GPU-memory, speedup,
resumable-training or app/UI-integration acceptance in this worker milestone.
Independent final execution is recorded separately from source-only review.

`independent-review.json` binds the unchanged11 source files and an independent
41-test run (14.77seconds, zero failures/skips). Earlier duplicate reviewer runs
with a shared log and denied process inspection are excluded from acceptance.

`app-integration.json` supersedes the earlier preparation-only limitation: the
final native reader sampling path and separate native episode metadata parse
ran through the actual isolated application API. The job completed12 genuine
student updates, verified fresh reload, registered the student, and served the
exact package download. The six native episodes/three lineage groups contain
generated scenes, so this remains software integration evidence.95 focused
core/contract tests passed, including repeated cancellation of actual child
processes, source/receipt integrity, retained Linux wheel variant identities
and rejection of consistently truncated corpus lengths. No task-quality claim.


`student-pipeline.json` records a real follow-through of the same saved student:
native INT8 conversion followed by CPU inference on three explicitly selected
observations from the immutable generated dataset. The independent review checks
artifact ancestry, packed model identity, all 1,800 finite output coordinates and
reset repeatability. Source student/dataset/configuration were preserved. This
exercises the new independent persistent model/reader environments through the
local application; no physical task quality, calibration, GPU or Isaac acceptance
is implied. Installation history still includes the separately retained timeouts.
