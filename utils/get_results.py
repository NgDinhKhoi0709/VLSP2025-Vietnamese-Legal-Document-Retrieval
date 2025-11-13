import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

def load_predictions(path: Path, topk: int) -> Dict[int, List[str]]:
    """Load predictions, returning mapping *qid ➜ list[id]* (length ≤ *topk*).

    Điều chỉnh: chỉ xét đúng topk phần tử đầu (có thể trùng), rồi loại trùng
    ngay trong đó, không lấy thêm để bù đủ.
    """
    with path.open(encoding="utf-8") as f:
        preds = json.load(f)

    grouped: Dict[int, List[tuple[float, str]]] = defaultdict(list)

    for item in preds:
        qid = item["qid"]
        # Xử lý các định dạng chunk khác nhau
        if "top_chunks" in item and isinstance(item["top_chunks"], list):
            chunks = item["top_chunks"]
        elif "chunks" in item and isinstance(item["chunks"], list):
            chunks = item["chunks"]
        else:
            chunks = [item]

        for ch in chunks:
            chunk_val = ch.get("chunk_id") or ch.get("id") or ch.get("doc_id")
            if chunk_val is None:
                continue
            base_id = str(chunk_val).split("_")[0]
            score = ch.get("score", 0.0)
            grouped[qid].append((score, base_id))

    result: Dict[int, List[str]] = {}
    for qid, lst in grouped.items():
        # 1) sort giảm dần
        lst.sort(key=lambda x: x[0], reverse=True)
        # 2) chỉ lấy đúng topk raw entries
        raw_topk = lst[:topk]
        # 3) loại trùng ngay trong raw_topk
        unique_ids = []
        seen = set()
        for _, doc_id in raw_topk:
            if doc_id not in seen:
                unique_ids.append(doc_id)
                seen.add(doc_id)
        result[qid] = unique_ids

    return result

def main():
    # Input file path
    input_path = Path("results/public_test/ensemble_results.json")
    # Output file path
    output_path = Path("results/public_test/results.json")
    # Number of top results to keep
    topk = 3  # You can adjust this value as needed

    # Load and process predictions
    predictions = load_predictions(input_path, topk)

    # Convert to the requested output format
    output_data = [
        {
            "qid": qid,
            "relevant_laws": [int(law_id) for law_id in law_ids]
        }
        for qid, law_ids in predictions.items()
    ]

    # Sort by qid for consistency
    output_data.sort(key=lambda x: x["qid"])

    # Write the output
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()