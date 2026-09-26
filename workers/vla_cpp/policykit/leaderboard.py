from __future__ import annotations

import html
import json
from pathlib import Path

from .metrics import comparisons


def render(results_path: Path, destination: Path) -> Path:
    payload = json.loads(results_path.read_text())
    selections = payload.get("deployment_selection", {})
    selection_rows = "".join(
        f"<li><strong>{html.escape(model)}</strong>: {html.escape(str(choice.get('selected_preset') or 'blocked'))} — {html.escape(choice.get('reason', ''))}</li>"
        for model, choice in sorted(selections.items())
    ) or "<li>Selection appears after completed benchmark results are saved.</li>"
    rows = []
    for item, comparison in zip(payload["results"], comparisons(payload["results"])):
        rate = item.get("success_rate")
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in [
            item["model"], item["preset"], item["status"],
            f"{item['artifact_bytes']:,}" if item.get("artifact_bytes") else "—",
            f"{rate:.1%}" if rate is not None else "—",
            f"{item['p50_ms']:.1f}" if item.get("p50_ms") is not None else "—",
            f"{item['p90_ms']:.1f}" if item.get("p90_ms") is not None else "—",
            f"{item['peak_rss_mb']:.1f}" if item.get("peak_rss_mb") is not None else "—",
            f"{item['quantize_seconds']:.2f}" if item.get("quantize_seconds") is not None else "—",
            f"{item['conversion_seconds']:.2f}" if item.get("conversion_seconds") is not None else "—",
            f"{item['module_wall_seconds']:.2f}" if item.get("module_wall_seconds") is not None else "—",
            f"{item['load_ms']:.1f}" if item.get("load_ms") is not None else "—",
            f"{comparison['speedup']:.2f}x" if comparison['speedup'] is not None else "—",
            f"{comparison['success_drop_pp']:.2f}" if comparison['success_drop_pp'] is not None else "—",
            comparison['quantization_break_even_calls'] if comparison['quantization_break_even_calls'] is not None else "—",
            item.get("error") or "",
        ]) + "</tr>")
    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>PolicyKit leaderboard</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:3rem auto;padding:0 1rem}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:.6rem;text-align:left}}th{{background:#f4f4f4}}.note{{background:#fff6d8;padding:1rem}}</style></head>
<body><h1>PolicyKit VLA quantization leaderboard</h1><p class='note'><strong>Target:</strong> {html.escape(payload['machine_label'])}. These CPU results are not RTX-laptop or Jetson results. Scores compare quantization within the same model; cross-model results are not normalized.</p>
<p>Run: {html.escape(payload['run_id'])} · vla.cpp {html.escape(payload['runtime_tag'])}</p>
<h2>Deployment selection</h2><p>Per-model selection uses its BF16 reference and the configured quality gate; it does not compare architectures.</p><ul>{selection_rows}</ul>
<p>Quantization includes artifact writing and auditing. Module wall time also includes uncached conversion and hashing. Speedup uses median inference latency; break-even excludes conversion and loading. Missing timings remain unknown.</p>
<table><thead><tr><th>Model</th><th>Precision</th><th>Status</th><th>Artifact bytes</th><th>LIBERO success</th><th>p50 ms</th><th>p90 ms</th><th>Loaded RSS MB</th><th>Quantize s</th><th>Conversion s</th><th>Module wall s</th><th>Load ms</th><th>Speedup</th><th>Success drop pp</th><th>Break-even calls</th><th>Failure</th></tr></thead><tbody>{''.join(rows)}</tbody></table></body></html>"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page)
    return destination
