# Cloud training and quantization verification

The following live checks ran on 2026-09-26/27 with the application hosted on
xbox-360 and isolated SkyPilot workers on Google Cloud. They verify the cloud
implementation before its integration with the newer upstream Spatial,
checkpoint-recovery and diagnostics changes. Post-integration software checks
are recorded separately below. Private deployment identifiers, credentials and
raw machine inventories are excluded from the public repository.

## Completed GPU training

| Check | SmolVLA LoRA | ACT |
| --- | --- | --- |
| Worker | NVIDIA L4 | NVIDIA L4 |
| Dataset | SO-101 pickup | PushT |
| Optimizer steps / batch | 100 / 4 | 200 / 4 |
| Checkpoints published to GCS | 20, 40, 60, 80, 100 | 40, 80, 120, 160, 200 |
| First recorded held-out loss | 0.1149895681 at step 20 | 0.8020 at step 20 |
| Last recorded held-out loss | 0.0789211897 at step 100 | 0.5242 at step 200 |
| Fresh-process action reload | Passed | Exact agreement (maximum difference 0.0) |
| Application-host job directory | 720 KiB | 509,627 bytes |

SmolVLA was launched through the actual training button in Chrome. ACT exercised
the separate native LeRobot adapter. The curves were non-monotonic and establish
held-out imitation loss only, not convergence or robot task success. SmolVLA
reported 229.3 worker seconds and 1,672,154,112 bytes peak allocated GPU memory.
ACT used action chunks of 20. Its pinned dataset revision was
`7628202a2180972f291ba1bc6723834921e72c19`.

Each job completed final verification, published its five checkpoints in private
GCS and cleaned up its owned GPU cluster. Checkpoints were not copied onto the
Mac or application host. An explicit SmolVLA download check streamed and hashed
60,909,280 bytes across 14 artifact files without saving model weights locally.

## Completed SmolVLA quantization

The actual **Start quantization** button selected the trained step-100 checkpoint
and dispatched an L4 worker. It downloaded the source checkpoint from GCS,
merged the adapters, exported GGUF and packed 112 language matrices to Q4_0.
Vision and action weights retained their protected precision. Q4 was an explicit
test candidate; the integrated application defaults to Q8 and labels Q4 as
experimental, following upstream task-quality evidence.

| Check | Result |
| --- | ---: |
| Source training payload | 60,904,447 bytes |
| Quantized GGUF | 1,070,308,800 bytes |
| Final bundle payload | 1,070,384,130 bytes |
| Native output | 1,600 finite values (50 × 32 padded channels; 6 real channels) |
| Native CPU forward | 11.132 seconds, one call, two threads |

A freshly built `vla_predict_check` loaded the packed model and ran one forward
pass using deterministic synthetic image, token, state and noise inputs. This
tests actual native execution, not task success. The worker fixes preserve
F32/F16/BF16 token embedding storage types and install the required native build
dependencies. Earlier failed attempts remain in the private application history.

GGUF SHA-256:
`d64ebd5847efbbb8efbc7d166b4b50a202804a255f1204908e7a0407197fa19d`.
The complete bundle was streamed through the application HTTP endpoint on Xbox
loopback in 207.0 seconds. All 15 TAR members, including the manifest, passed
size/hash verification. No weight or TAR cache was created on either host.
The Mac SSH tunnel was tested separately and was slower; the complete 1 GB
hash check was performed on Xbox loopback.

GCS read-ahead is bounded to 16 MiB with at most 1 MiB pipe/HTTP chunks.
Interleaved 16 MiB range checks took 20.68/19.43 seconds with 1 MiB GCS requests
and 3.25/3.30 seconds with 16 MiB read-ahead. These are observed transport rates,
not a throughput guarantee.

## Model coverage and limits

All 15 model choices observed in KiteML have registered training routes with
pinned sources and compatibility checks. All 13 native LeRobot profiles passed
actual configuration parsing and Linux dependency resolution. See
[configuration records](native-config-validation.json) and the
[dependency audit](native-dependency-resolution.md). SmolVLA and ACT completed
GPU training; other profiles have contract/configuration evidence only.

Psi-Zero A100 attempts exhausted regional capacity before worker setup. No
optimizer steps or checkpoints were produced; all owned requests/resources were
cleaned up. This does not establish GPU validation of Psi-Zero.

GGUF quantization currently supports SmolVLA. Other native checkpoints support
training, resume and download. Isaac Sim runners and Spatial evaluation arrived
from upstream during integration; this cloud verification does not claim a live
Isaac rollout or Spatial hardware acceptance.

## Software verification

Before upstream integration: 811 core tests passed (two optional-environment
skips), 191 desktop/mobile browser checks passed, Ruff and formatting passed,
and the production client built using the declared Node 24.21.0 runtime.
These checks are separate from the real GPU evidence above.

The merged inventory, model/method and dataset/camera admission checks also
accepted the saved metadata and full recipes from the real SmolVLA step-100 and
ACT step-200 outputs (plus earlier step-20/40 checkpoint descriptors). This is
metadata compatibility with existing artifacts, not a new GPU reload.


Post-integration checks against upstream `7672e3e` (including the cloud-run collector):

| Check | Result |
| --- | --- |
| Core, fresh Python 3.14.7 with frozen dependencies | 988 passed, 2 optional/platform skips |
| Real GCS SDK read-ahead regression (separate environment) | 1 passed |
| SmolVLA/native/Psi training worker contracts | 175 passed, 8 optional-environment skips |
| Native quantization/Spatial worker contracts | 252 passed, 4 platform/vendor skips |
| Desktop/mobile application browser tests | 221 passed |
| Dedicated desktop/mobile diagnostics | 18 passed |
| Standalone cloud monitor and credential redaction, Python 3.12 | 12 passed |
| `/firebird` prefixed export smoke | 1 passed |
| Ruff, formatting, generated schema/client and TypeScript | Passed |
| Production and prefixed exports, Node 24.21.0 | Passed |

Recovery regressions include repeated cloud resumes with preserved camera/model
recipes, checkpoint-owned worker selection, stale/torn indexes and fallback from
corrupt newer descriptors. The integration does not redeploy the running Xbox
release or repeat the GPU runs above. Hosted CI and platform/GPU acceptance remain
separate checks.
