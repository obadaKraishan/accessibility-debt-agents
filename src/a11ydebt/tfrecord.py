"""
Dependency-free readers for AndroidControl's TFRecord shards and
AndroidAccessibilityForest protobuf blobs.

WHY THIS EXISTS
---------------
The AndroidControl README tells you to install TensorFlow and to clone
google-deepmind/android_env. Both work, and this project uses them when
they are available. But TensorFlow is a 500 MB dependency that regularly
fails to install on Apple Silicon and in constrained environments, and the
open GitHub issue asking how to decode `accessibility_trees` (issue #2120)
has no answer on it. So the pipeline ships its own readers, implemented
against the protobuf wire format, and falls back to them automatically.

Both paths produce identical output. phase4_validation cross-checks them
against each other on a sample when TensorFlow is present.

    from a11ydebt.tfrecord import iter_records, parse_example, parse_forest

    for blob in iter_records(shard_path):
        example = parse_example(blob)
        trees = example["accessibility_trees"]        # list[bytes]
        forest = parse_forest(trees[0])               # list[Window]

FIELD NUMBERS
-------------
Taken from android_env/proto/a11y/*.proto. They are part of the wire
contract, so they are stable across android-env releases; if a future
release renumbers them, phase0's proto check and phase4's cross-check will
both fail loudly rather than produce wrong numbers silently.
"""

from __future__ import annotations

import gzip
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

__all__ = [
    "iter_records", "parse_example", "parse_forest", "resolve_click",
    "Node", "Window", "WINDOW_TYPES", "TruncatedShardError",
]


class TruncatedShardError(ValueError):
    """Raised when a shard ends mid-record.

    Carries `records_read` so the caller can report how much was recovered.
    A truncated shard is usually an interrupted download: re-running the
    download script resumes it. Callers should warn loudly rather than
    silently analysing a partial corpus.
    """

    def __init__(self, message: str, records_read: int = 0) -> None:
        super().__init__(message)
        self.records_read = records_read

# --------------------------------------------------------------------------
# protobuf wire format
# --------------------------------------------------------------------------

_WIRE_VARINT, _WIRE_64, _WIRE_LEN, _WIRE_32 = 0, 1, 2, 5


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 70:
            raise ValueError("varint too long")


def _iter_fields(buf: bytes) -> Iterator[tuple[int, int, Any]]:
    """Yield (field_number, wire_type, value). Length-delimited yields bytes."""
    pos = 0
    while pos < len(buf):
        key, pos = _read_varint(buf, pos)
        field_no, wire = key >> 3, key & 0x07
        if wire == _WIRE_VARINT:
            value, pos = _read_varint(buf, pos)
        elif wire == _WIRE_64:
            value = buf[pos:pos + 8]
            pos += 8
        elif wire == _WIRE_LEN:
            length, pos = _read_varint(buf, pos)
            value = buf[pos:pos + length]
            pos += length
        elif wire == _WIRE_32:
            value = buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield field_no, wire, value


def _packed_varints(buf: bytes) -> list[int]:
    out, pos = [], 0
    while pos < len(buf):
        value, pos = _read_varint(buf, pos)
        out.append(value)
    return out


def _zigzag_to_signed(value: int, bits: int = 32) -> int:
    """int32 fields are stored as plain varints; large values are negatives."""
    limit = 1 << (bits - 1)
    if value >= (1 << bits):
        # int64 negative stored in 10 bytes
        return value - (1 << 64)
    return value - (1 << bits) if value >= limit else value


# --------------------------------------------------------------------------
# TFRecord container
# --------------------------------------------------------------------------

def iter_records(path: str | Path, compression: str = "GZIP") -> Iterator[bytes]:
    """
    Yield raw record payloads from a TFRecord file.

    Layout per record:
        uint64  length          (little endian)
        uint32  masked crc32c of the length bytes   <- not verified here
        bytes   payload
        uint32  masked crc32c of the payload        <- not verified here

    CRCs are skipped deliberately: crc32c is not in the standard library,
    the files come from a trusted HTTPS source with a byte-count check at
    download time, and verifying would triple the read cost of 46 GB.
    """
    path = Path(path)
    opener = gzip.open if compression.upper() == "GZIP" else open
    count = 0
    with opener(path, "rb") as fh:
        while True:
            try:
                header = fh.read(8)
            except (EOFError, gzip.BadGzipFile, zlib.error) as exc:
                raise TruncatedShardError(
                    f"{path.name}: compressed stream ends mid-record after "
                    f"{count:,} records ({exc.__class__.__name__}).", count
                ) from exc
            if not header:
                return
            if len(header) < 8:
                raise TruncatedShardError(
                    f"{path.name}: truncated record header after {count:,} records.",
                    count)
            (length,) = struct.unpack("<Q", header)
            try:
                fh.read(4)                   # length crc
                payload = fh.read(length)
                fh.read(4)                   # payload crc
            except (EOFError, gzip.BadGzipFile, zlib.error) as exc:
                raise TruncatedShardError(
                    f"{path.name}: stream ends mid-record after {count:,} records "
                    f"({exc.__class__.__name__}).", count
                ) from exc
            if len(payload) < length:
                raise TruncatedShardError(
                    f"{path.name}: truncated record payload after {count:,} records.",
                    count)
            count += 1
            yield payload


# --------------------------------------------------------------------------
# tf.train.Example
# --------------------------------------------------------------------------
# Example  { Features features = 1 }
# Features { map<string, Feature> feature = 1 }
# map entry{ string key = 1; Feature value = 2 }
# Feature  { BytesList bytes_list = 1; FloatList float_list = 2;
#            Int64List int64_list = 3 }
# BytesList{ repeated bytes value = 1 }
# Int64List{ repeated int64 value = 1 }

def _parse_feature(buf: bytes) -> list:
    for field_no, _wire, value in _iter_fields(buf):
        if field_no == 1:                                    # bytes_list
            return [v for f, _w, v in _iter_fields(value) if f == 1]
        if field_no == 2:                                    # float_list
            out = []
            for f, w, v in _iter_fields(value):
                if f != 1:
                    continue
                if w == _WIRE_LEN:
                    out += [struct.unpack("<f", v[i:i + 4])[0] for i in range(0, len(v), 4)]
                elif w == _WIRE_32:
                    out.append(struct.unpack("<f", v)[0])
            return out
        if field_no == 3:                                    # int64_list
            out = []
            for f, w, v in _iter_fields(value):
                if f != 1:
                    continue
                out += _packed_varints(v) if w == _WIRE_LEN else [v]
            return out
    return []


def parse_example(blob: bytes) -> dict[str, list]:
    """Decode a tf.train.Example into {feature_name: list_of_values}."""
    out: dict[str, list] = {}
    for field_no, _wire, features_buf in _iter_fields(blob):
        if field_no != 1:
            continue
        for f2, _w2, entry in _iter_fields(features_buf):
            if f2 != 1:
                continue
            key, feature = None, []
            for f3, _w3, value in _iter_fields(entry):
                if f3 == 1:
                    key = value.decode("utf-8", "replace")
                elif f3 == 2:
                    feature = _parse_feature(value)
            if key is not None:
                out[key] = feature
    return out


# --------------------------------------------------------------------------
# AndroidAccessibilityForest
# --------------------------------------------------------------------------

WINDOW_TYPES = {
    0: "UNKNOWN_TYPE", 1: "TYPE_APPLICATION", 2: "TYPE_INPUT_METHOD",
    3: "TYPE_SYSTEM", 4: "TYPE_ACCESSIBILITY_OVERLAY",
    5: "TYPE_SPLIT_SCREEN_DIVIDER", 6: "TYPE_MAGNIFICATION_OVERLAY",
    7: "TYPE_WINDOW_CONTROL",
}


@dataclass
class Node:
    """One AndroidAccessibilityNodeInfo, with the fields this study uses."""

    unique_id: int = -1
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0
    class_name: str = ""
    content_description: str = ""
    hint_text: str = ""
    package_name: str = ""
    text: str = ""
    view_id_resource_name: str = ""
    window_id: int = -1
    is_checkable: bool = False
    is_checked: bool = False
    is_clickable: bool = False
    is_editable: bool = False
    is_enabled: bool = False
    is_focusable: bool = False
    is_long_clickable: bool = False
    is_scrollable: bool = False
    is_selected: bool = False
    is_visible_to_user: bool = False
    child_ids: list[int] = field(default_factory=list)
    depth: int = 0
    labeled_by_id: int = -1
    label_for_id: int = -1
    drawing_order: int = 0
    tooltip_text: str = ""

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom

    @property
    def area(self) -> int:
        return max(0, self.right - self.left) * max(0, self.bottom - self.top)


@dataclass
class Window:
    window_id: int = -1
    layer: int = 0
    title: str = ""
    window_type: str = "UNKNOWN_TYPE"
    is_active: bool = False
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0
    nodes: list[Node] = field(default_factory=list)

    def node_map(self) -> dict[int, Node]:
        """
        Nodes keyed by unique_id.

        Two traps this guards against, both verified against the real data:
        list index does NOT equal unique_id (observed order 0,1,2,3,4,5,7,6),
        and unique_id restarts per window so it is only unique WITHIN a window.
        Callers that need a forest-wide key must use (window_id, unique_id).
        """
        return {n.unique_id: n for n in self.nodes}


def _parse_rect(buf: bytes) -> tuple[int, int, int, int]:
    rect = [0, 0, 0, 0]
    for field_no, _wire, value in _iter_fields(buf):
        if 1 <= field_no <= 4 and isinstance(value, int):
            rect[field_no - 1] = _zigzag_to_signed(value)
    return tuple(rect)  # type: ignore[return-value]


_NODE_STR = {3: "class_name", 4: "content_description", 5: "hint_text",
             6: "package_name", 7: "text", 10: "view_id_resource_name",
             31: "tooltip_text"}
_NODE_BOOL = {12: "is_checkable", 13: "is_checked", 14: "is_clickable",
              15: "is_editable", 16: "is_enabled", 17: "is_focusable",
              19: "is_long_clickable", 21: "is_scrollable", 22: "is_selected",
              23: "is_visible_to_user"}
_NODE_INT = {1: "unique_id", 11: "window_id", 27: "depth",
             28: "labeled_by_id", 29: "label_for_id", 30: "drawing_order"}


def _parse_node(buf: bytes) -> Node:
    node = Node()
    for field_no, wire, value in _iter_fields(buf):
        if field_no == 2 and wire == _WIRE_LEN:
            node.left, node.top, node.right, node.bottom = _parse_rect(value)
        elif field_no in _NODE_STR and wire == _WIRE_LEN:
            setattr(node, _NODE_STR[field_no], value.decode("utf-8", "replace"))
        elif field_no in _NODE_BOOL and wire == _WIRE_VARINT:
            setattr(node, _NODE_BOOL[field_no], bool(value))
        elif field_no in _NODE_INT and wire == _WIRE_VARINT:
            setattr(node, _NODE_INT[field_no], _zigzag_to_signed(value))
        elif field_no == 25:                                  # child_ids
            if wire == _WIRE_LEN:
                node.child_ids.extend(_zigzag_to_signed(v) for v in _packed_varints(value))
            elif wire == _WIRE_VARINT:
                node.child_ids.append(_zigzag_to_signed(value))
    return node


def _parse_window(buf: bytes) -> Window:
    window = Window()
    for field_no, wire, value in _iter_fields(buf):
        if field_no == 1 and wire == _WIRE_LEN:
            window.left, window.top, window.right, window.bottom = _parse_rect(value)
        elif field_no == 3 and wire == _WIRE_VARINT:
            window.window_id = _zigzag_to_signed(value)
        elif field_no == 4 and wire == _WIRE_VARINT:
            window.layer = _zigzag_to_signed(value)
        elif field_no == 5 and wire == _WIRE_LEN:
            window.title = value.decode("utf-8", "replace")
        elif field_no == 6 and wire == _WIRE_VARINT:
            window.window_type = WINDOW_TYPES.get(value, f"TYPE_{value}")
        elif field_no == 8 and wire == _WIRE_VARINT:
            window.is_active = bool(value)
        elif field_no == 11 and wire == _WIRE_LEN:            # tree
            for f2, w2, tree_value in _iter_fields(value):
                if f2 == 1 and w2 == _WIRE_LEN:               # nodes
                    window.nodes.append(_parse_node(tree_value))
    return window


def parse_forest(blob: bytes) -> list[Window]:
    """Decode an AndroidAccessibilityForest into a list of Windows."""
    windows: list[Window] = []
    for field_no, wire, value in _iter_fields(blob):
        if field_no == 1 and wire == _WIRE_LEN:
            windows.append(_parse_window(value))
    return windows


# --------------------------------------------------------------------------
# hit testing (used by RQ3)
# --------------------------------------------------------------------------

def resolve_click(
    windows: list[Window],
    x: float,
    y: float,
    fallback_to_clickable_ancestor: bool = True,
) -> tuple[Node | None, Window | None]:
    """
    Resolve a click coordinate to the element it hit.

    AndroidControl actions carry only x and y; there is no element reference
    in the data, so which element a trajectory touched must be inferred. On
    the parsed episodes, 99.92% of clicks and long presses resolved; of those,
    43.71% were reassigned by step 4 and 18.42% found no clickable ancestor
    (outputs/manuscript/manuscript_numbers.txt, ReviewerChecks section). The rule:

      1. Consider only nodes with is_visible_to_user and positive area.
      2. Restrict to the window with the highest layer that contains the point.
      3. Take the deepest such node; break ties on drawing_order, then on
         the smallest area, which is the most specific target.
      4. If that node is not clickable, walk up the parent chain to the
         nearest clickable ancestor; if none is clickable, keep that node.

    Returns (node, window), or (None, None) if nothing contains the point.
    """
    candidates: list[tuple[int, Window, Node]] = []
    for window in windows:
        for node in window.nodes:
            if not node.is_visible_to_user or node.area <= 0:
                continue
            if node.contains(x, y):
                candidates.append((window.layer, window, node))
    if not candidates:
        return None, None

    top_layer = max(c[0] for c in candidates)
    top = [(w, n) for layer, w, n in candidates if layer == top_layer]
    window, node = max(top, key=lambda wn: (wn[1].depth, wn[1].drawing_order, -wn[1].area))

    if fallback_to_clickable_ancestor and not node.is_clickable:
        parents: dict[int, int] = {}
        for candidate in window.nodes:
            for child in candidate.child_ids:
                parents[child] = candidate.unique_id
        by_id = window.node_map()
        seen, current = set(), node.unique_id
        while current in parents and current not in seen:
            seen.add(current)
            current = parents[current]
            parent = by_id.get(current)
            if parent is None:
                break
            if parent.is_clickable:
                return parent, window
    return node, window
