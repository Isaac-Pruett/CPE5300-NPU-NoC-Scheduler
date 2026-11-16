"""Load and snapshot hardware configurations for the analytical backends."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_ROOT = Path(__file__).resolve().parent / "configs"
CONFIG_FILES = ("system.json", "network.yml", "remote_memory.json")
TOPOLOGIES = {"FullyConnected", "Ring", "Switch"}


def available_configs() -> list[Path]:
    """List complete bundled directories, excluding workload and draft folders."""
    return sorted(
        path
        for path in CONFIG_ROOT.iterdir()
        if path.is_dir() and all((path / name).is_file() for name in CONFIG_FILES)
    )


def resolve_config_dir(value: str | Path) -> Path:
    """Bare bundled names are portable; other paths are relative to the caller."""
    path = Path(value).expanduser()
    if path.name == str(value) and (CONFIG_ROOT / path).is_dir():
        path = CONFIG_ROOT / path
    return path.resolve()


def _number(value: object, label: str, *, positive: bool = False) -> None:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        constraint = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be a finite {constraint} number")


@dataclass(frozen=True)
class HardwareConfig:
    directory: Path
    num_ranks: int
    # Keep the validated bytes so later edits cannot change the simulated inputs.
    files: tuple[tuple[str, bytes], ...]
    description: str

    def write_to(self, directory: Path) -> None:
        for name, content in self.files:
            (directory / name).write_bytes(content)

    def summary(self) -> dict:
        return {
            "name": self.directory.name,
            "source_directory": str(self.directory),
            "num_ranks": self.num_ranks,
            "description": self.description,
            "sha256": {
                name: hashlib.sha256(content).hexdigest()
                for name, content in self.files
            },
        }


def load_config(
    value: str | Path,
    *,
    num_ranks: int | None = None,
    backend: str = "congestion-aware",
) -> HardwareConfig:
    """Reject incompatible inputs before creating a run or invoking ASTRA.

    Validate the runner's fixed-duration workload contract and common analytical
    config fields. Backend-specific optional fields remain ASTRA's responsibility.
    """
    directory = resolve_config_dir(value)
    if not directory.is_dir():
        raise ValueError(f"configuration directory does not exist: {directory}")
    missing = [name for name in CONFIG_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"{directory}: missing required files: {', '.join(missing)}")
    files = []
    parsed = {}
    for name in CONFIG_FILES:
        path = directory / name
        try:
            content = path.read_bytes()
            document = (
                yaml.safe_load(content)
                if name.endswith(".yml")
                else json.loads(content)
            )
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise ValueError(f"cannot read {path}: {exc}") from exc
        if not isinstance(document, dict):
            raise ValueError(f"{path}: expected a mapping/object")  # noqa: TRY004 -- invalid file content
        files.append((name, content))
        parsed[name] = document

    network = parsed["network.yml"]
    dimensions = network.get("topology")
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError("network.yml: topology must be a non-empty list")
    for field in ("npus_count", "bandwidth", "latency"):
        values = network.get(field)
        if not isinstance(values, list) or len(values) != len(dimensions):
            raise ValueError(
                f"network.yml: {field} must have one entry per topology dimension"
            )
    for index, topology in enumerate(dimensions):
        if not isinstance(topology, str) or topology not in TOPOLOGIES:
            raise ValueError(f"network.yml: unsupported topology {topology!r}")
        count = network["npus_count"][index]
        if type(count) is not int or count < 1:
            raise ValueError(
                "network.yml: npus_count entries must be positive integers"
            )
        if count == 1 and topology != "FullyConnected":
            raise ValueError(
                "network.yml: singleton dimensions require FullyConnected topology"
            )
        if backend == "congestion-aware" and topology == "Ring" and count == 2:
            raise ValueError(
                "network.yml: a two-node Ring is incompatible with congestion-aware; use FullyConnected"
            )
        _number(
            network["bandwidth"][index],
            f"network.yml: bandwidth[{index}] (GB/s)",
            positive=True,
        )
        _number(network["latency"][index], f"network.yml: latency[{index}] (ns)")
    actual_ranks = math.prod(network["npus_count"])
    if num_ranks is not None and actual_ranks != num_ranks:
        raise ValueError(
            f"network.yml: topology has {actual_ranks} ranks but workload requires {num_ranks}"
        )

    system = parsed["system.json"]
    if system.get("scheduling-policy") not in ("FIFO", "LIFO", "EXPLICIT"):
        raise ValueError(
            "system.json: scheduling-policy must be FIFO, LIFO, or EXPLICIT"
        )
    if (
        type(system.get("roofline-enabled")) is not int
        or system["roofline-enabled"] != 0
    ):
        raise ValueError(
            "system.json: roofline-enabled must be 0 for fixed-duration workload traces"
        )
    for field in ("endpoint-delay", "local-mem-bw", "peak-perf"):
        if field in system:
            _number(
                system[field],
                f"system.json: {field}",
                positive=field != "endpoint-delay",
            )
    for field in (
        "all-reduce-implementation",
        "all-gather-implementation",
        "reduce-scatter-implementation",
        "all-to-all-implementation",
    ):
        if field in system:
            algorithms = system[field]
            if (
                not isinstance(algorithms, list)
                or len(algorithms) != len(dimensions)
                or not all(isinstance(item, str) and item for item in algorithms)
            ):
                raise ValueError(
                    f"system.json: {field} must have one algorithm per topology dimension"
                )
    if system.get("collective-optimization", "localBWAware") not in (
        "localBWAware",
        "baseline",
    ):
        raise ValueError(
            "system.json: collective-optimization must be localBWAware or baseline"
        )
    memory = parsed["remote_memory.json"]
    if not isinstance(memory.get("memory-type"), str) or not memory["memory-type"]:
        raise ValueError("remote_memory.json: memory-type must be a non-empty string")

    description = "Custom analytical configuration"
    metadata_path = directory / "profile.json"
    if metadata_path.exists():
        try:
            content = metadata_path.read_bytes()
            metadata = json.loads(content)
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read {metadata_path}: {exc}") from exc
        if not isinstance(metadata, dict) or not isinstance(
            metadata.get("description"), str
        ):
            raise ValueError(
                f"{metadata_path}: expected an object with a description string"
            )
        description = metadata["description"]
        files.append(("profile.json", content))
    return HardwareConfig(directory, actual_ranks, tuple(files), description)
