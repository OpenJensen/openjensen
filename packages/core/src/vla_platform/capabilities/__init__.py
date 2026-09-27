from vla_platform.contracts import Capability


def registry(
    native_configured: bool = False,
    training_configured: bool = False,
    act_export_configured: bool = False,
    cloud_configured: bool = False,
    native_quantization_configured: bool = False,
) -> list[Capability]:
    configured = {"Quantize", "Evaluate", "Run"} if native_configured or cloud_configured else set()
    if native_quantization_configured:
        configured.add("Quantize")
    if training_configured or cloud_configured:
        configured.add("Fine-tune")
    return [
        Capability(
            stage="Run",
            operation="policy.export",
            status="untested" if act_export_configured else "planned",
            description=(
                "ACT FP32 export removes only the training VAE and checks synthetic CPU parity. "
                "Requires a complete local native checkpoint and a separately configured worker. "
                "No task success, calibration or GPU speed claim."
            ),
        ),
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
                status="untested" if stage in configured else "planned",
                description=(
                    "Native ACT 4/8-bit packing and fresh CPU reload are configured; "
                    "generated action drift only, no simulation, quality or speedup claim."
                    if stage == "Quantize" and native_quantization_configured
                    else "Execution worker configured; target support evidence is unregistered. "
                    "Each job performs runtime preflight."
                    if stage in configured
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
