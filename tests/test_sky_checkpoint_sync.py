import json
import sys
from types import ModuleType, SimpleNamespace

from vla_platform.lifecycle import sky_checkpoint_sync


def test_selective_transfer_downloads_index_and_only_unseen_snapshots(
    tmp_path, monkeypatch, capsys
):
    cluster = "fb-aabbccddeeff-11223344"
    remote = f"/server/cache/{cluster}/1-firebird-training"
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    entries = [
        {
            "name": f"checkpoint-{step:06d}",
            "step": step,
            "manifest_sha256": str(step // 5) * 64,
            "file_bytes": 100,
        }
        for step in (5, 10)
    ]
    (index_dir / "index.json").write_text(json.dumps({"checkpoints": entries}))
    receipts = tmp_path / "receipts.json"
    receipts.write_text(json.dumps({"checkpoint-000005": "1" * 64}))
    downloads = []

    def download(paths, *, remote_machine_prefix, local_machine_prefix):
        assert remote_machine_prefix == remote
        assert local_machine_prefix.endswith(f"/{cluster}/1-firebird-training")
        downloads.extend(paths)
        path = paths[0]
        return {
            path: str(index_dir if path.endswith("checkpoint-index") else tmp_path / "download")
        }

    class Body:
        def __init__(self, *, cluster_name, job_ids):
            assert cluster_name == cluster and job_ids == ["1"]

        def model_dump_json(self):
            return "{}"

    def request(method, route, **kwargs):
        assert (method, route) == ("POST", "/download_logs")
        return "request"

    sky = ModuleType("sky")
    sky.get = lambda _: {"1": remote}
    client = ModuleType("sky.client")
    client.common = SimpleNamespace(download_logs_from_api_server=download)
    server = ModuleType("sky.server")
    server.common = SimpleNamespace(
        make_authenticated_request=request, get_request_id=lambda _: "id"
    )
    requests = ModuleType("sky.server.requests")
    requests.payloads = SimpleNamespace(ClusterJobsDownloadLogsBody=Body)
    for name, module in {
        "sky": sky,
        "sky.client": client,
        "sky.server": server,
        "sky.server.requests": requests,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "argv", ["sync.py", cluster, str(receipts), "final"])
    sky_checkpoint_sync.main()
    assert downloads == [
        remote + "/checkpoint-index",
        remote + "/checkpoint-snapshots/checkpoint-000010",
        remote + "/final-output",
    ]
    output = capsys.readouterr().out
    assert "FIREBIRD_CHECKPOINT=" in output and '"step": 10' in output
    assert '"step": 5' not in output
