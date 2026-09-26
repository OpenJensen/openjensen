import json
from pathlib import Path

from policykit.leaderboard import render


def test_render_includes_failed_rows(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"run_id": "r", "machine_label": "test machine", "runtime_tag": "v0.3.0", "results": [{"model": "pi0", "preset": "bf16", "status": "failed", "artifact_bytes": None, "success_rate": None, "p50_ms": None, "p90_ms": None, "error": "out of memory"}]}))
    page = render(results, tmp_path / "index.html")
    assert "out of memory" in page.read_text()
    assert "not RTX-laptop or Jetson" in page.read_text()


def test_render_shows_timing_and_quality_comparison(tmp_path):
    results = tmp_path/'results.json'
    results.write_text(json.dumps({'run_id':'r','machine_label':'cpu','runtime_tag':'test','results':[
        {'model':'smolvla','preset':'bf16','status':'complete','success_rate':.8,'p50_ms':100},
        {'model':'smolvla','preset':'q4','status':'complete','success_rate':.75,'p50_ms':50,
         'quantize_seconds':7.25,'conversion_seconds':3.5,'module_wall_seconds':11,'load_ms':1000}]}))
    page = render(results,tmp_path/'index.html').read_text()
    assert '<th>Quantize s</th>' in page
    assert '<th>Success drop pp</th>' in page
    assert '<td>7.25</td>' in page
    assert '<td>2.00x</td>' in page
    assert '<td>5.00</td>' in page
    assert '<td>145</td>' in page
