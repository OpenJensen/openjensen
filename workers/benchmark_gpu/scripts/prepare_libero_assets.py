"""Cache pinned LIBERO assets and write noninteractive configuration."""

import argparse
import importlib.util
import json
from pathlib import Path

REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"


def main():
    import yaml
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.find_spec("libero")
    if spec is None:
        raise RuntimeError("Install hf-libero==0.1.3 first")
    benchmark = Path(next(iter(spec.submodule_search_locations))) / "libero"
    for folder in ("bddl_files", "init_files"):
        if not (benchmark / folder).is_dir():
            raise FileNotFoundError(benchmark / folder)
    assets = snapshot_download("lerobot/libero-assets", repo_type="dataset", revision=REVISION)
    # hf-libero's arena loader also resolves assets relative to its package.
    # Link the isolated environment to the exact snapshot; never replace data.
    asset_link = benchmark / "assets"
    if asset_link.is_symlink():
        if asset_link.resolve() != Path(assets).resolve():
            raise FileExistsError("Installed LIBERO asset link points to another snapshot")
    elif asset_link.exists():
        if any(asset_link.iterdir()):
            raise FileExistsError("Installed LIBERO asset directory is nonempty")
        asset_link.rmdir()
        asset_link.symlink_to(assets, target_is_directory=True)
    else:
        asset_link.symlink_to(assets, target_is_directory=True)
    args.config_dir.mkdir(parents=True, exist_ok=True)
    target = args.config_dir / "config.yaml"
    config = {
        "assets": assets,
        "benchmark_root": str(benchmark),
        "bddl_files": str(benchmark / "bddl_files"),
        "init_states": str(benchmark / "init_files"),
        "datasets": str(args.config_dir.resolve() / "unused-datasets"),
    }
    if target.exists() and yaml.safe_load(target.read_text()) != config:
        raise FileExistsError("Existing LIBERO config differs; choose a new directory")
    target.write_text(yaml.safe_dump(config))
    (args.config_dir / "assets.json").write_text(
        json.dumps(
            {
                "repository": "lerobot/libero-assets",
                "revision": REVISION,
                "path": assets,
            },
            indent=2,
        )
        + "\n"
    )
    print(target.resolve())


if __name__ == "__main__":
    main()
