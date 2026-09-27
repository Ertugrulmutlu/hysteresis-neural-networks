"""Generate paired seed configs and optionally execute them sequentially."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

from src.config import load_experiment_config

ALLOWED_PAIR_DIFFERENCES = {"train.scenario", "logging.run_name", "logging.save_dir"}


def _manifest_protocol_view(manifest: dict[str, Any]) -> dict[str, Any]:
    """Compare scientific/run identity without binding manifests to one Python executable."""
    fields = ("seed", "scenario", "config_path", "expected_run_directory")
    return {"seeds": manifest.get("seeds"),
            "entries": [{key: entry.get(key) for key in fields} for entry in manifest.get("entries", [])]}


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result = {}
    for key, child in value.items():
        result.update(_flatten(child, f"{prefix}.{key}" if prefix else key))
    return result


def validate_generated_pair(sabc: dict[str, Any], sbac: dict[str, Any]) -> None:
    if (sabc["train"]["scenario"], sbac["train"]["scenario"]) != ("SABC", "SBAC"):
        raise ValueError("Generated pair must be ordered SABC/SBAC")
    a, b = _flatten(sabc), _flatten(sbac)
    differences = [key for key in sorted(set(a) | set(b)) if a.get(key) != b.get(key) and key not in ALLOWED_PAIR_DIFFERENCES]
    if differences:
        raise ValueError(f"Generated pair differs outside allowlist: {differences}")


def generate_seed_config(base: dict[str, Any], seed: int, scenario: str) -> dict[str, Any]:
    config = json.loads(json.dumps(base)); config["experiment"]["seed"] = int(seed); config["train"]["scenario"] = scenario
    name = str(config["logging"]["run_name"]); marker = "_seed"
    prefix, separator, suffix = name.rpartition(marker)
    if not separator or "_norm" not in suffix:
        raise ValueError(f"Run name must contain _seed<value>_norm: {name}")
    norm_suffix = suffix[suffix.index("_norm"):]
    config["logging"]["run_name"] = f"{prefix}{marker}{seed}{norm_suffix}"
    config["logging"]["overwrite"] = False
    return config


def build_manifest_entries(base_sabc: dict[str, Any], base_sbac: dict[str, Any], seeds: Sequence[int], config_dir: Path):
    entries = []
    for seed in seeds:
        configs = (generate_seed_config(base_sabc, seed, "SABC"), generate_seed_config(base_sbac, seed, "SBAC"))
        validate_generated_pair(*configs)
        for config in configs:
            scenario = config["train"]["scenario"]; path = config_dir / f"seed{seed}_{scenario.lower()}.yaml"
            run_dir = Path(config["logging"]["save_dir"]) / config["logging"]["run_name"]
            entries.append({"seed": seed, "scenario": scenario, "config_path": str(path),
                            "expected_run_directory": str(run_dir), "command": [sys.executable, "-m", "src.train", str(path)],
                            "config": config})
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--base-sabc", required=True); parser.add_argument("--base-sbac", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True); parser.add_argument("--generated-config-dir", required=True)
    parser.add_argument("--manifest-out", required=True); parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true"); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.execute and args.dry_run: raise ValueError("Choose either --dry-run or --execute")
    config_dir, manifest_path = Path(args.generated_config_dir), Path(args.manifest_out)
    entries = build_manifest_entries(load_experiment_config(args.base_sabc), load_experiment_config(args.base_sbac), args.seeds, config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        path = Path(entry["config_path"])
        entry["config"]["logging"]["overwrite"] = bool(args.overwrite)
        generated_config = entry.pop("config")
        rendered = yaml.safe_dump(generated_config, sort_keys=False)
        if path.exists() and load_experiment_config(path) != generated_config and not args.overwrite:
            raise FileExistsError(f"Generated config differs and exists: {path}")
        if not path.exists() or args.overwrite: path.write_text(rendered, encoding="utf-8")
    manifest = {"entries": entries, "seeds": list(args.seeds)}; manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_manifest=json.dumps(manifest,indent=2)
    if manifest_path.exists() and not args.overwrite:
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _manifest_protocol_view(existing_manifest) != _manifest_protocol_view(manifest):
            raise FileExistsError(f"Manifest differs and exists: {manifest_path}")
    if not manifest_path.exists() or args.overwrite: manifest_path.write_text(rendered_manifest,encoding="utf-8")
    for entry in entries:
        print(subprocess.list2cmdline(entry["command"]))
    if not args.execute:
        return
    for entry in entries:
        run_dir = Path(entry["expected_run_directory"])
        if run_dir.exists() and any(run_dir.iterdir()):
            if args.skip_existing: continue
            if not args.overwrite: raise FileExistsError(f"Run directory exists: {run_dir}")
        subprocess.run(entry["command"], check=True)


if __name__ == "__main__":
    main()
