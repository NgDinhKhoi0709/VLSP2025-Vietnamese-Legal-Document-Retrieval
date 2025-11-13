import argparse
import json
import os
from typing import Any


def fix_chunk_id(chunk_id: str) -> str:
    """Collapse duplicated document id in a chunk id.

    Examples
    --------
    >>> fix_chunk_id("56908_56908_0")
    '56908_0'
    >>> fix_chunk_id("1088_1088_24")
    '1088_24'
    >>> fix_chunk_id("123_456_0")
    '123_456_0'  # unchanged because the prefix is not duplicated
    """
    if not chunk_id:
        return chunk_id

    parts = chunk_id.split("_")
    # Only transform strings that look like <doc>_<doc>_<segment>[ _<subsegment>...]
    if len(parts) >= 3 and parts[0] == parts[1]:
        # keep the first part (document id) and everything after the duplicated part
        return "_".join([parts[0]] + parts[2:])
    return chunk_id


def apply_recursively(obj: Any) -> None:
    """Recursively walk a JSON-like structure and update `chunk_id` fields in place."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "chunk_id" and isinstance(value, str):
                obj[key] = fix_chunk_id(value)
            else:
                apply_recursively(value)
    elif isinstance(obj, list):
        for item in obj:
            apply_recursively(item)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix duplicated document prefix in chunk_id fields of a JSON file."
    )
    parser.add_argument("input", help="Path to the input JSON file")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to save the corrected JSON file (defaults to overwrite input)",
        default=None
    )
    args = parser.parse_args()

    in_path: str = args.input
    out_path: str = args.output or args.input

    # Read JSON (supports both compact and pretty-printed files)
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Mutate in place
    apply_recursively(data)

    # Ensure directory exists for output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # Write back using UTF-8 and pretty indent for readability
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Chunk IDs fixed and written to {out_path}")


if __name__ == "__main__":
    main() 