from vla_platform.contracts import Capability


def registry() -> list[Capability]:
    return [
        Capability(
            stage="Dataset",
            operation="dataset.inspect",
            status="untested",
            description=(
                "Public HF and allowed local LeRobot metadata only; "
                "no semantic or media validation. Target evidence awaits CAP-001."
            ),
        ),
        *[
            Capability(
                stage=stage,
                operation=operation,
                status="planned",
                description=description,
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
