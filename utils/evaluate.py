from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set

BETA = 2  # F2-score
BETA_SQ = BETA ** 2

ROOT = Path(__file__).resolve().parents[1]  # project root
GT_PATH = ROOT / "data" / "processed" / "test.json"


def fbeta_score(pred: Set[str], gold: Set[str], beta_sq: int = BETA_SQ) -> float:
    """Compute F-beta for a single sample.

    Parameters
    ----------
    pred: set[str]
        Predicted document ids.
    gold: set[str]
        Ground-truth document ids.
    beta_sq: int, default 4
        Beta squared (\(\beta^2\)). For F2 this is 4.
    """
    if not pred and not gold:
        return 1.0  # Perfect when both are empty (unlikely here but for completeness)

    if not pred:
        return 0.0

    tp = len(pred & gold)
    fp = len(pred) - tp
    fn = len(gold) - tp

    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp
    return 0.0 if denominator == 0 else numerator / denominator


def load_ground_truth(path: Path = GT_PATH) -> Dict[int, Set[str]]:
    """Load ground-truth mapping *qid ➜ set(relevant_law_ids)*."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {item["qid"]: {str(law_id) for law_id in item["relevant_laws"]} for item in data}

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

def compute_macro_f2(gt: Dict[int, Set[str]], pred: Dict[int, List[str]]) -> float:
    """Compute macro F2 across all queries present in *gt*.

    If a *qid* is missing from *pred*, an empty prediction is used.
    """
    scores = []
    for qid, gold in gt.items():
        predicted = set(pred.get(qid, []))
        score = fbeta_score(predicted, gold)
        scores.append(score)
    return sum(scores) / len(scores)


def main():
    """Run evaluation using predefined variables instead of CLI arguments."""
    # ----- User-configurable variables -----
    TOPK = 3  # Số id tối đa giữ lại cho mỗi truy vấn
    PRED_PATHS = {
        "test_rerank_bgem3_base": ROOT / "results" / "test" / "test_rerank_bgem3_base.json",
        "ENSEMBLE test_rerank_bgem3_base": ROOT / "results" / "test" / "product_rank_ensemble_test_rerank_bgem3_base_bm25.json",
        "test_rerank_bgem3_finetune": ROOT / "results" / "test" / "test_rerank_bgem3_finetune.json",

        "ENSEMBLE test_rerank_bgem3_finetune": ROOT / "results" / "test" / "product_rank_ensemble_test_rerank_bgem3_finetune_bm25.json",
        
        "test_rerank_gte_base": ROOT / "results" / "test" / "test_rerank_gte_base.json",
        "ENSEMBLE test_rerank_gte_base": ROOT / "results" / "test" / "product_rank_ensemble_test_rerank_gte_base_bm25.json",
        "test_rerank_gte_finetune": ROOT / "results" / "test" / "test_rerank_gte_finetune.json",
        "ENSEMBLE test_rerank_gte_finetune": ROOT / "results" / "test" / "product_rank_ensemble_test_rerank_gte_finetune_bm25.json",

        # "test_rerank_qwen3_8b": ROOT / "results" / "lamlai" / "test" / "test_rerank_qwen3_8b.json",
        # "ENSEMBLE test_rerank_qwen3_8b_bm25_test": ROOT / "results" / "lamlai" / "test" / "ensemble_test_rerank_qwen3_8b_bm25_test.json",
        # "ENSEMBLE bge_512_bm25_test": ROOT / "results" / "lamlai" / "test" / "ensemble_bge_512_bm25_test.json",
        # "ENSEMBLE product_rank_ensemble_bge_512_bm25_test": ROOT / "results" / "lamlai" / "test" / "product_rank_ensemble_bge_512_bm25_test.json",
        # "ENSEMBLE product_ensemble_bge_512_bm25_test": ROOT / "results" / "lamlai" / "test" / "product_ensemble_bge_512_bm25_test.json",

    }

    # ---------------------------------------

    gt = load_ground_truth()

    print(f"Loaded ground-truth for {len(gt)} queries from {GT_PATH}")
    print(f"Using TOPK = {TOPK}\n")

    for name, path in PRED_PATHS.items():
        preds = load_predictions(path, TOPK)
        macro_f2 = compute_macro_f2(gt, preds)
        print(f"{name:>6}: {macro_f2:.4f}")


if __name__ == "__main__":
    main()
