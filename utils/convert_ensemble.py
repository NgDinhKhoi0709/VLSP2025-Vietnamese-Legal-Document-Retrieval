import json
from pathlib import Path

def convert(input_path: str, output_path: str | None = None, topk: int | None = None):
    """Convert ensemble_bm25_+_bkai_v1.json to desired format.

    Args:
        input_path: path to source json.
        output_path: save path, default same folder with *_laws.json suffix.
        topk: keep only the first K ids (after merging & sorting). If None, keep all.

    Steps:
    1. For each query, merge chunk scores that share the same document id (part before the underscore).
    2. Sum scores when ids collide.
    3. Sort ids by their (summed) score in ascending order (lower score comes first).
    4. Slice to TOPK if provided.
    5. Create a list of dictionaries matching the required schema and save as JSON.
    """

    input_path = str(input_path)
    if output_path is None:
        input_name = Path(input_path).stem  # ensemble_bm25_+_bkai_v1
        output_path = Path(input_path).with_name(f"{input_name}_laws.json")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted: list[dict] = []

    for entry in data:
        qid = entry["qid"]
        chunk_items = entry.get("top_chunks", [])

        # Deduplicate by law id keeping first occurrence within sorted chunks
        # Sort chunks by ascending score to ensure correct order
        chunk_items_sorted = sorted(chunk_items, key=lambda x: x["score"], reverse=True)

        seen: set[str] = set()
        unique_ids: list[int] = []
        for item in chunk_items_sorted:
            law_id_str = item["chunk_id"].split("_")[0]
            if law_id_str in seen:
                continue
            seen.add(law_id_str)
            unique_ids.append(int(law_id_str))
            if topk is not None and len(unique_ids) >= topk:
                break

        sorted_ids = unique_ids

        converted.append({"qid": int(qid), "relevant_laws": sorted_ids})

    # Persist result
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Saved converted results to {output_path}")


if __name__ == "__main__":
    # ==== User adjustable variables ====
    INPUT_PATH = "../results/public_test/ensemble_bm25_+_bkai_v1.json"  # <-- chỉnh đường dẫn nếu cần
    OUTPUT_PATH = "../results/public_test/results.json"  # hoặc đặt thành "path/to/save.json"
    TOPK = 5  # đặt None để lấy tất cả, hoặc số nguyên để lấy TOPK id
    # ====================================

    convert(INPUT_PATH, OUTPUT_PATH, TOPK)
