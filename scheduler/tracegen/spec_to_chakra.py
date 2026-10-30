"""Convert the speculative-decoding event IR to per-rank Chakra traces."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from speculative.event_model import Event, EventTrace, EventType


try:
    from chakra.schema.protobuf.et_def_pb2 import (
        COMM_RECV_NODE,
        COMM_SEND_NODE,
        COMP_NODE,
        AttributeProto,
        GlobalMetadata,
        Node,
    )
    from chakra.src.third_party.utils.protolib import encodeMessage
except ImportError as exc:
    raise ImportError(
        "Unable to import Chakra. Build the project image and run "
        "'docker compose run --rm scheduler python -m tracegen.spec_to_chakra ...'. "
        "The image provides matching generated bindings and protobuf runtime."
    ) from exc


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "traces" / "speculative_events.json"
DEFAULT_OUTPUT_PREFIX = Path(__file__).resolve().parents[1] / "traces" / "spec_decode"
CHAKRA_VERSION = "0.0.4"


def _numeric_tag_map(events: Iterable[Event]) -> dict[str, int]:
    """Assign stable nonzero uint32 tags to the IR's readable string tags."""
    tags = sorted(
        {
            event.tag
            for event in events
            if event.event_type in {EventType.SEND, EventType.RECV}
            and event.tag is not None
        }
    )
    return {tag: index for index, tag in enumerate(tags, start=1)}


def _convert_event(event: Event, tag_map: dict[str, int], comm_priority: int) -> Node:
    """Convert one validated IR event to a Chakra node."""
    node = Node(id=event.event_id, name=event.name)
    node.data_deps.extend(event.dependencies)
    node.attr.append(AttributeProto(name="is_cpu_op", bool_val=False))
    if event.phase is not None:
        node.attr.append(AttributeProto(name="phase", string_val=event.phase))

    if event.event_type is EventType.COMPUTE:
        node.type = COMP_NODE
        node.duration_micros = event.duration_us
        return node

    node.type = COMM_SEND_NODE if event.event_type is EventType.SEND else COMM_RECV_NODE
    node.attr.extend(
        [
            AttributeProto(name="comm_src", uint32_val=event.src),
            AttributeProto(name="comm_dst", uint32_val=event.dst),
            AttributeProto(name="comm_size", uint64_val=event.bytes),
            AttributeProto(name="comm_tag", uint32_val=tag_map[event.tag]),
            AttributeProto(name="comm_priority", uint32_val=comm_priority),
            AttributeProto(name="spec_tag", string_val=event.tag),
        ]
    )
    return node


def convert_trace(
    trace: EventTrace,
    output_prefix: str | Path,
    comm_priority: int = 0,
) -> list[Path]:
    """Write one Chakra execution-trace file for each rank."""
    trace.validate()
    if not 0 <= comm_priority <= 0xFFFFFFFF:
        raise ValueError("comm_priority must fit in a uint32")

    prefix = Path(output_prefix)
    if prefix.suffix == ".et":
        raise ValueError("output prefix must not include a .et suffix")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    tag_map = _numeric_tag_map(trace.events)
    output_paths: list[Path] = []

    for rank in range(trace.num_ranks):
        rank_events = sorted(
            (event for event in trace.events if event.rank == rank),
            key=lambda event: event.event_id,
        )
        if not rank_events:
            raise ValueError(f"rank {rank} has no events")

        output_path = Path(f"{prefix}.{rank}.et")
        metadata = GlobalMetadata(
            version=CHAKRA_VERSION,
            attr=[
                AttributeProto(name="source", string_val="scheduler_event_ir"),
                AttributeProto(name="source_schema", string_val=trace.schema_version),
                AttributeProto(name="rank", uint32_val=rank),
                AttributeProto(name="num_ranks", uint32_val=trace.num_ranks),
            ],
        )

        with output_path.open("wb") as output_file:
            encodeMessage(output_file, metadata)
            for event in rank_events:
                encodeMessage(
                    output_file,
                    _convert_event(event, tag_map, comm_priority),
                )

        output_paths.append(output_path)

    return output_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert speculative event JSON to per-rank Chakra traces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input speculative event JSON",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output prefix; rank and .et suffix are appended",
    )
    parser.add_argument(
        "--comm-priority",
        type=int,
        default=0,
        help="ASTRA-sim communication priority assigned to point-to-point nodes",
    )
    return parser.parse_args()


def main() -> None:
    """Load the IR and emit one Chakra file per rank."""
    args = parse_args()
    trace = EventTrace.read_json(args.input)
    output_paths = convert_trace(trace, args.output_prefix, args.comm_priority)

    print("Speculative event IR converted to Chakra")
    print(f"  input: {args.input}")
    for rank, output_path in enumerate(output_paths):
        print(f"  rank {rank}: {output_path}")


if __name__ == "__main__":
    main()
