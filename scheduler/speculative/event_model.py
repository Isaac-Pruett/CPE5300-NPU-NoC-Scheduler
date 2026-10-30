"""Simulator-independent event model for speculative-decoding workloads."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any


class EventType(str, Enum):
    """Operations supported by the first speculative-decoding milestone."""

    COMPUTE = "COMPUTE"
    SEND = "SEND"
    RECV = "RECV"


@dataclass(slots=True)
class Event:
    """One compute or point-to-point communication operation."""

    event_id: int
    event_type: EventType
    rank: int
    name: str
    dependencies: list[int] = field(default_factory=list)
    duration_us: int | None = None
    src: int | None = None
    dst: int | None = None
    bytes: int | None = None
    tag: str | None = None
    phase: str | None = None

    def validate(self, num_ranks: int) -> None:
        """Validate fields that do not require the complete event graph."""
        if self.event_id < 0:
            raise ValueError(f"event_id must be non-negative: {self.event_id}")
        if not 0 <= self.rank < num_ranks:
            raise ValueError(f"event {self.event_id} has invalid rank {self.rank}")
        if not self.name:
            raise ValueError(f"event {self.event_id} must have a name")
        if self.phase not in {None, "prefill", "decode"}:
            raise ValueError(f"event {self.event_id} has invalid phase {self.phase}")
        if self.event_id in self.dependencies:
            raise ValueError(f"event {self.event_id} cannot depend on itself")
        if len(self.dependencies) != len(set(self.dependencies)):
            raise ValueError(f"event {self.event_id} has duplicate dependencies")

        if self.event_type is EventType.COMPUTE:
            if self.duration_us is None or self.duration_us < 0:
                raise ValueError(
                    f"compute event {self.event_id} requires non-negative duration_us"
                )
            return

        if self.event_type not in {EventType.SEND, EventType.RECV}:
            raise ValueError(f"unsupported event type: {self.event_type}")
        if self.src is None or not 0 <= self.src < num_ranks:
            raise ValueError(f"event {self.event_id} has invalid src {self.src}")
        if self.dst is None or not 0 <= self.dst < num_ranks:
            raise ValueError(f"event {self.event_id} has invalid dst {self.dst}")
        if self.src == self.dst:
            raise ValueError(f"event {self.event_id} must communicate between ranks")
        if self.bytes is None or self.bytes <= 0:
            raise ValueError(f"event {self.event_id} requires a positive byte count")
        if not self.tag:
            raise ValueError(f"event {self.event_id} requires a communication tag")
        if self.event_type is EventType.SEND and self.rank != self.src:
            raise ValueError(f"send event {self.event_id} must execute on src rank")
        if self.event_type is EventType.RECV and self.rank != self.dst:
            raise ValueError(f"recv event {self.event_id} must execute on dst rank")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        result: dict[str, Any] = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "rank": self.rank,
            "name": self.name,
            "dependencies": list(self.dependencies),
        }
        for key in ("duration_us", "src", "dst", "bytes", "tag", "phase"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        """Construct an event from decoded JSON."""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            rank=data["rank"],
            name=data["name"],
            dependencies=list(data.get("dependencies", [])),
            duration_us=data.get("duration_us"),
            src=data.get("src"),
            dst=data.get("dst"),
            bytes=data.get("bytes"),
            tag=data.get("tag"),
            phase=data.get("phase"),
        )


@dataclass(slots=True)
class EventTrace:
    """A complete multi-rank speculative-decoding event graph."""

    num_ranks: int
    events: list[Event]
    parameters: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    def validate(self) -> None:
        """Validate rank-local DAGs and point-to-point communication pairs."""
        if self.num_ranks <= 0:
            raise ValueError("num_ranks must be positive")
        if not self.events:
            raise ValueError("trace must contain at least one event")

        events_by_id: dict[int, Event] = {}
        for event in self.events:
            event.validate(self.num_ranks)
            if event.event_id in events_by_id:
                raise ValueError(f"duplicate event_id: {event.event_id}")
            events_by_id[event.event_id] = event

        for event in self.events:
            for dependency_id in event.dependencies:
                dependency = events_by_id.get(dependency_id)
                if dependency is None:
                    raise ValueError(
                        f"event {event.event_id} references missing dependency "
                        f"{dependency_id}"
                    )
                if dependency.rank != event.rank:
                    raise ValueError(
                        f"event {event.event_id} has cross-rank dependency "
                        f"{dependency_id}; use a SEND/RECV pair instead"
                    )

        visit_state: dict[int, int] = {}

        def visit(event_id: int) -> None:
            state = visit_state.get(event_id, 0)
            if state == 1:
                raise ValueError(f"dependency cycle detected at event {event_id}")
            if state == 2:
                return
            visit_state[event_id] = 1
            for dependency_id in events_by_id[event_id].dependencies:
                visit(dependency_id)
            visit_state[event_id] = 2

        for event_id in events_by_id:
            visit(event_id)

        def communication_key(event: Event) -> tuple[int | None, int | None, str | None, int | None]:
            return (event.src, event.dst, event.tag, event.bytes)

        sends = Counter(
            communication_key(event)
            for event in self.events
            if event.event_type is EventType.SEND
        )
        receives = Counter(
            communication_key(event)
            for event in self.events
            if event.event_type is EventType.RECV
        )
        if sends != receives:
            missing_receives = sends - receives
            missing_sends = receives - sends
            raise ValueError(
                "unmatched communication events: "
                f"missing receives={dict(missing_receives)}, "
                f"missing sends={dict(missing_sends)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable trace after validating it."""
        self.validate()
        return {
            "schema_version": self.schema_version,
            "num_ranks": self.num_ranks,
            "parameters": self.parameters,
            "events": [event.to_dict() for event in self.events],
        }

    def write_json(self, output_path: str | Path) -> Path:
        """Validate and write the trace as deterministic, readable JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventTrace:
        """Construct and validate a trace from decoded JSON."""
        trace = cls(
            schema_version=data.get("schema_version", "1.0"),
            num_ranks=data["num_ranks"],
            parameters=dict(data.get("parameters", {})),
            events=[Event.from_dict(event) for event in data["events"]],
        )
        trace.validate()
        return trace

    @classmethod
    def read_json(cls, input_path: str | Path) -> EventTrace:
        """Read and validate a trace from disk."""
        data = json.loads(Path(input_path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("event trace JSON must contain an object")
        return cls.from_dict(data)
