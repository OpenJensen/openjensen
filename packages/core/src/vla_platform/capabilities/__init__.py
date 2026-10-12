from vla_platform.contracts import Capability


def registry(
    native_configured: bool = False,
    training_configured: bool = False,
    act_export_configured: bool = False,
    cloud_configured: bool = False,
    native_quantization_configured: bool = False,
    native_distillation_configured: bool = False,
    native_replay_configured: bool = False,
) -> list[Capability]:
    configured = {"Quantize", "Evaluate", "Run"} if native_configured or cloud_configured else set()
    if native_quantization_configured:
        configured.add("Quantize")
    if native_replay_configured:
        configured.add("Run")
    if native_distillation_configured:
        configured.add("Distill")
    if training_configured or cloud_configured:
        configured.add("Fine-tune")
    return [
        Capability(
            stage="Run",
            operation="policy.export",
            status="untested" if act_export_configured else "planned",
            description=(
                "Export a complete ACT checkpoint as an FP32 inference package. "
                "Requires a separately configured export worker. "
                "Evaluate the exported policy before using it for a task."
            ),
        ),
        Capability(
            stage="Dataset",
            operation="dataset.inspect",
            status="untested",
            description=(
                "Inspect public Hugging Face and allowed local LeRobot dataset metadata. "
                "Dataset contents and media availability depend on the source."
            ),
        ),
        *[
            Capability(
                stage=stage,
                operation=operation,
                status="untested" if stage in configured else "planned",
                description=(
                    "Local ACT teacher to ACT256 student distillation is configured; "
                    "explicit recorded-coordinate compatibility and three disjoint episode "
                    "and lineage partitions are required. Evaluate the student on your task "
                    "before deployment."
                    if stage == "Distill" and native_distillation_configured
                    else "Packed ACT CPU observation replay is configured; exact outputs and "
                    "reset repeatability are measured without advancing a simulated environment."
                    if stage == "Run" and native_replay_configured and not native_configured
                    else "Native ACT 4/8-bit packing and fresh CPU reload are configured; "
                    "compare generated-input action differences before evaluating your task."
                    if stage == "Quantize" and native_quantization_configured
                    else "Execution worker configured. "
                    "Each job checks runtime compatibility before starting."
                    if stage in configured
                    else description
                ),
                support=[],
            )
            for stage, operation, description in [
                (
                    "Fine-tune",
                    "policy.finetune",
                    "Configure a compatible training worker to fine-tune a policy.",
                ),
                (
                    "Distill",
                    "policy.distill",
                    "ACT to ACT256 distillation requires a separately configured local worker.",
                ),
                (
                    "Quantize",
                    "policy.quantize",
                    "Configure a compatible quantization worker to create a packed policy.",
                ),
                (
                    "Evaluate",
                    "policy.evaluate",
                    "Closed-loop simulation requires a compatible runtime.",
                ),
                (
                    "Run",
                    "policy.run",
                    "Configure a compatible runtime to run an imported policy.",
                ),
            ]
        ],
    ]
