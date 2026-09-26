from vla_platform.contracts import Capability


def registry(
    native_configured: bool = False, training_configured: bool = False
) -> list[Capability]:
    available = {"Quantize", "Evaluate", "Run"} if native_configured else set()
    if training_configured:
        available.add("Fine-tune")
    return [
        Capability(
            stage="Dataset",
            operation="dataset.inspect",
            status="available",
            support=[],  # Target coverage is unknown until CAP-001 supplies reviewed evidence.
            description=(
                "Public HF and allowed local LeRobot metadata only; "
                "no semantic or media validation."
            ),
        ),
        *[
            Capability(
                stage=stage,
                operation=operation,
                status="available" if stage in available else "planned",
                description=(
                    "Native worker configured; each job performs runtime preflight."
                    if stage in available
                    else description
                ),
                support=[],
            )
            for stage, operation, description in [
                ("Fine-tune", "policy.finetune", "Native training recipe integration is planned."),
                (
                    "Distill",
                    "policy.distill",
                    "A supported teacher/student pair is not selected yet.",
                ),
                (
                    "Quantize",
                    "policy.quantize",
                    "Packed conversion and reload remain unimplemented.",
                ),
                (
                    "Evaluate",
                    "policy.evaluate",
                    "Closed-loop simulation requires a compatible runtime.",
                ),
                (
                    "Run",
                    "policy.run",
                    "Exact tested policy export and native execution are planned.",
                ),
            ]
        ],
    ]
