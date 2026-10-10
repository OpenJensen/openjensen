# Worker setup

Run each worker's installation commands in its own directory with the Python
version named in its guide.

| Setup | Guide |
| --- | --- |
| Parquet reader | [Install and preview](_cpu_readers/README.md) |
| SmolVLA training | [Install, train and resume](smolvla_qlora/README.md) |
| ACT export | [Install and export](act_optimizer/README.md) |
| ACT distillation | [Prepare data and run](policy_distillation/README.md) |
| Local ACT CPU runtimes | [Install and configure](local_cpu/README.md) |
| Packed weights | [Install and quantize](firebird_quant/README.md) |
| Native ACT packing | [Run the package worker](firebird_quant/NATIVE_ACT.md) |
| SmolVLA GGUF | [Prepare and convert](vla_cpp/README.md) |
| GPU comparison | [Prepare and run](benchmark_gpu/README.md) |
| Psi-Zero | [Install and run](psi0/README.md) |
| Muose scorer | [Install and score](decision/README.md) |
| Isaac | [Build and record](isaac_sim/README.md) |
| SkyPilot | [Configure and launch](skypilot/README.md) |
| Teaching | [Configure and record](teaching/README.md) |

For application runtime configuration, use the [workflow setup](../docs/policy-workflow.md).
