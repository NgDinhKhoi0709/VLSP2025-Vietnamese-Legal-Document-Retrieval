import json
import os
from collections import defaultdict

def ensemble_topk_global_minmax(file_paths, weights=None, K=10, output_path='ensemble_results.json'):
    # 1) Compute global min/max per file
    stats = {}
    for p in file_paths:
        all_scores = []
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for rec in data:
                all_scores.extend([c['score'] for c in rec.get('top_chunks', [])])
        mn, mx = (min(all_scores), max(all_scores)) if all_scores else (0.0, 1.0)
        stats[p] = (mn, mx)

    # default weights = 1.0
    if weights is None:
        weights = {}
    # 2) Aggregate scaled scores
    agg = defaultdict(lambda: defaultdict(float))
    for p in file_paths:
        mn, mx = stats[p]
        w = weights.get(p, 1.0)
        with open(p, 'r', encoding='utf-8') as f:
            for rec in json.load(f):
                qid = rec['qid']
                for c in rec.get('top_chunks', []):
                    raw = c['score']
                    scaled = (raw - mn) / (mx - mn) if mx > mn else 0.0
                    agg[qid][c['chunk_id']] += scaled * w

    # 3) Take TOP K and prepare output
    output_list = []
    topk_results = {}
    for qid, scores in agg.items():
        topk = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:K]
        topk_results[qid] = topk
        output_list.append({
            "qid": qid,
            "top_chunks": [
                {"chunk_id": cid, "score": sc}
                for cid, sc in topk
            ]
        })

    # 4) Write JSON file
    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(output_list, out_f, ensure_ascii=False, indent=2)

    return topk_results

def _load_score_map(file_path: str) -> dict:
    """Load a results JSON file into { qid: { chunk_id: score } } map."""
    score_map = defaultdict(dict)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for rec in data:
            qid = rec['qid']
            for c in rec.get('top_chunks', []):
                score_map[qid][c['chunk_id']] = c['score']
    return score_map

def ensemble_pair_product(model_path: str,
                          bm25_path: str,
                          model_weight: float,
                          bm25_weight: float,
                          K: int,
                          output_path: str):
    model_scores = _load_score_map(model_path)
    bm25_scores = _load_score_map(bm25_path)

    output_list = []
    combined = {}
    all_qids = set(model_scores.keys()) | set(bm25_scores.keys())

    for qid in all_qids:
        model_map = model_scores.get(qid, {})
        bm25_map = bm25_scores.get(qid, {})
        intersection_keys = set(model_map.keys()) & set(bm25_map.keys())

        scores_accumulator = []
        
        # Process intersection chunks with product scoring
        for cid in intersection_keys:
            sc = (model_weight * model_map[cid]) * (bm25_weight * bm25_map[cid])
            scores_accumulator.append((cid, sc))
        
        # Add model-only chunks (chunks in model but not in BM25)
        model_only_keys = set(model_map.keys()) - intersection_keys
        for cid in model_only_keys:
            sc = model_weight * model_map[cid]
            scores_accumulator.append((cid, sc))
        
        # Add BM25-only chunks (chunks in BM25 but not in model)
        bm25_only_keys = set(bm25_map.keys()) - intersection_keys
        for cid in bm25_only_keys:
            sc = bm25_weight * bm25_map[cid]
            scores_accumulator.append((cid, sc))

        topk = sorted(scores_accumulator, key=lambda x: x[1], reverse=True)[:K]
        combined[qid] = topk
        output_list.append({
            "qid": qid,
            "top_chunks": [
                {"chunk_id": cid, "score": sc} for cid, sc in topk
            ]
        })

    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(output_list, out_f, ensure_ascii=False, indent=2)

    return combined

def ensemble_pair_product_rank(model_path: str,
                               bm25_path: str,
                               model_weight: float,
                               bm25_weight: float,
                               K: int,
                               output_path: str):
    # Load raw score map for model and bm25
    model_scores = _load_score_map(model_path)
    bm25_scores = _load_score_map(bm25_path)

    # Build rank map from model file: { qid: { chunk_id: rank_index (1-based) } }
    model_rank_map = defaultdict(dict)
    with open(model_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for rec in data:
            qid = rec['qid']
            for idx, c in enumerate(rec.get('top_chunks', []), start=1):
                model_rank_map[qid][c['chunk_id']] = idx

    output_list = []
    combined = {}
    all_qids = set(model_scores.keys()) | set(bm25_scores.keys())

    for qid in all_qids:
        model_map = model_scores.get(qid, {})
        bm25_map = bm25_scores.get(qid, {})
        rank_map = model_rank_map.get(qid, {})
        intersection_keys = set(model_map.keys()) & set(bm25_map.keys())

        scores_accumulator = []
        
        # Process intersection chunks with product scoring and rank factor
        for cid in intersection_keys:
            rank = rank_map.get(cid)
            if rank:
                rank_factor = 1.0 / float(rank)
                sc = (model_weight * model_map[cid]) * (bm25_weight * bm25_map[cid]) * rank_factor
            else:
                # Use raw product without rank factor if rank is missing
                sc = (model_weight * model_map[cid]) * (bm25_weight * bm25_map[cid])
            scores_accumulator.append((cid, sc))
        
        # Add model-only chunks (chunks in model but not in BM25)
        model_only_keys = set(model_map.keys()) - intersection_keys
        for cid in model_only_keys:
            rank = rank_map.get(cid)
            if rank:
                rank_factor = 1.0 / float(rank)
                sc = (model_weight * model_map[cid]) * rank_factor
            else:
                # Use raw weighted score if rank is missing
                sc = model_weight * model_map[cid]
            scores_accumulator.append((cid, sc))
        
        # Add BM25-only chunks (chunks in BM25 but not in model)
        bm25_only_keys = set(bm25_map.keys()) - intersection_keys
        for cid in bm25_only_keys:
            sc = bm25_weight * bm25_map[cid]
            scores_accumulator.append((cid, sc))

        topk = sorted(scores_accumulator, key=lambda x: x[1], reverse=True)[:K]
        combined[qid] = topk
        output_list.append({
            "qid": qid,
            "top_chunks": [
                {"chunk_id": cid, "score": sc} for cid, sc in topk
            ]
        })

    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(output_list, out_f, ensure_ascii=False, indent=2)

    return combined

def ensemble_pair_product_bm25_rank(model_path: str,
                                    bm25_path: str,
                                    model_weight: float,
                                    bm25_weight: float,
                                    K: int,
                                    output_path: str):
    # Load raw score map for model and bm25
    model_scores = _load_score_map(model_path)
    bm25_scores = _load_score_map(bm25_path)

    # Build rank map from BM25 file: { qid: { chunk_id: rank_index (1-based) } }
    bm25_rank_map = defaultdict(dict)
    with open(bm25_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for rec in data:
            qid = rec['qid']
            for idx, c in enumerate(rec.get('top_chunks', []), start=1):
                bm25_rank_map[qid][c['chunk_id']] = idx

    output_list = []
    combined = {}
    all_qids = set(model_scores.keys()) | set(bm25_scores.keys())

    for qid in all_qids:
        model_map = model_scores.get(qid, {})
        bm25_map = bm25_scores.get(qid, {})
        rank_map = bm25_rank_map.get(qid, {})
        intersection_keys = set(model_map.keys()) & set(bm25_map.keys())

        scores_accumulator = []
        
        # Process intersection chunks with product scoring and BM25 rank factor
        for cid in intersection_keys:
            rank = rank_map.get(cid)
            if rank:
                rank_factor = 1.0 / float(rank)
                sc = (model_weight * model_map[cid]) * (bm25_weight * bm25_map[cid]) * rank_factor
            else:
                # Use raw product without rank factor if rank is missing
                sc = (model_weight * model_map[cid]) * (bm25_weight * bm25_map[cid])
            scores_accumulator.append((cid, sc))
        
        # Add model-only chunks (chunks in model but not in BM25)
        model_only_keys = set(model_map.keys()) - intersection_keys
        for cid in model_only_keys:
            sc = model_weight * model_map[cid]
            scores_accumulator.append((cid, sc))
        
        # Add BM25-only chunks (chunks in BM25 but not in model)
        bm25_only_keys = set(bm25_map.keys()) - intersection_keys
        for cid in bm25_only_keys:
            rank = rank_map.get(cid)
            if rank:
                rank_factor = 1.0 / float(rank)
                sc = (bm25_weight * bm25_map[cid]) * rank_factor
            else:
                # Use raw weighted score if rank is missing
                sc = bm25_weight * bm25_map[cid]
            scores_accumulator.append((cid, sc))

        topk = sorted(scores_accumulator, key=lambda x: x[1], reverse=True)[:K]
        combined[qid] = topk
        output_list.append({
            "qid": qid,
            "top_chunks": [
                {"chunk_id": cid, "score": sc} for cid, sc in topk
            ]
        })

    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(output_list, out_f, ensure_ascii=False, indent=2)

    return combined

def ensemble_multiple_models_with_bm25(
    results_dir='results/test',
    model_files=None,
    bm25_file='bm25_test.json',
    model_weight=1.1,
    bm25_weight=1.0,
    K=100,
    output_prefix='ensemble_'
):
    if model_files is None:
        # Auto-detect model files (exclude BM25 and existing ensemble files)
        model_files = []
        for file in os.listdir(results_dir):
            if (file.endswith('.json') and 
                'bm25' not in file and 
                'ensemble' not in file and
                file != bm25_file):
                model_files.append(file)
    
    bm25_path = os.path.join(results_dir, bm25_file)
    
    # Check if BM25 file exists
    if not os.path.exists(bm25_path):
        raise FileNotFoundError(f"BM25 file not found: {bm25_path}")
    
    results = {}
    
    for model_file in model_files:
        model_path = os.path.join(results_dir, model_file)
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found: {model_path}")
            continue
        
        # Extract model name for output file
        model_name = model_file.replace('.json', '').replace('_test', '')
        output_file = f"{output_prefix}{model_name}_bm25.json"
        output_path = os.path.join(results_dir, output_file)
        
        # Define file paths and weights
        file_paths = [model_path, bm25_path]
        weights = {
            model_path: model_weight,
            bm25_path: bm25_weight
        }
        
        # Perform ensemble
        result = ensemble_topk_global_minmax(
            file_paths=file_paths,
            weights=weights,
            K=K,
            output_path=output_path
        )
        
        results[output_file] = result
        print(f"✓ Completed: {output_file}\n")
    
    return results

def ensemble_pairs(
    results_dir: str,
    pairs: list,
    method: str = 'sum',
    K: int = 100,
):
    if method not in {'sum', 'product', 'product_rank', 'product_bm25_rank'}:
        raise ValueError("method must be either 'sum' or 'product'")

    results = {}

    for pair in pairs:
        model_file = pair['model']
        model_weight = float(pair.get('model_weight', 1.0))
        bm25_file = pair.get('bm25', 'bm25_test.json')
        bm25_weight = float(pair.get('bm25_weight', 1.0))
        # prefix method to output filename
        base_output_file = pair['output']
        method_prefix = f"{method}_"
        output_file = (
            base_output_file if base_output_file.startswith(method_prefix)
            else method_prefix + base_output_file
        )

        model_path = os.path.join(results_dir, model_file)
        bm25_path = os.path.join(results_dir, bm25_file)
        output_path = os.path.join(results_dir, output_file)

        if not os.path.exists(model_path):
            print(f"Warning: Model file not found: {model_path}")
            continue
        if not os.path.exists(bm25_path):
            print(f"Warning: BM25 file not found: {bm25_path}")
            continue

        if method == 'sum':
            file_paths = [model_path, bm25_path]
            weights = {model_path: model_weight, bm25_path: bm25_weight}
            result = ensemble_topk_global_minmax(
                file_paths=file_paths,
                weights=weights,
                K=K,
                output_path=output_path
            )
        elif method == 'product':
            result = ensemble_pair_product(
                model_path=model_path,
                bm25_path=bm25_path,
                model_weight=model_weight,
                bm25_weight=bm25_weight,
                K=K,
                output_path=output_path
            )
        elif method == 'product_rank':
            result = ensemble_pair_product_rank(
                model_path=model_path,
                bm25_path=bm25_path,
                model_weight=model_weight,
                bm25_weight=bm25_weight,
                K=K,
                output_path=output_path
            )
        else:  # product_bm25_rank
            result = ensemble_pair_product_bm25_rank(
                model_path=model_path,
                bm25_path=bm25_path,
                model_weight=model_weight,
                bm25_weight=bm25_weight,
                K=K,
                output_path=output_path
            )

        results[output_file] = result
        print(f"✓ Completed: {output_file}\n")

    return results

if __name__ == "__main__":
    print("=== CUSTOM ENSEMBLE WITH BM25 ===\n")

    # Choose ensemble method: 'sum' (minmax-scale + weighted sum) or 'product' (raw multiplication), 'product_rank' (raw multiplication + rank factor)
    ENSEMBLE_METHOD = 'product_rank'

    try:
        # Define custom pairs with per-model weights and optional bm25 variant/weight
        custom_pairs = [
            {
                'model': 'bge_512_private_test.json',
                'model_weight': 1.0,
                'bm25': 'bm25_512_private_test.json',
                'bm25_weight': 1.0,
                'output': 'ensemble_bge_512_bm25_private_test.json',
            }
        ]

        custom_results = ensemble_pairs(
            results_dir='results/private_test',
            pairs=custom_pairs,
            method=ENSEMBLE_METHOD,
            K=1000,
        )
    except Exception as e:
        print(f"Custom ensemble failed: {e}\n")

    print("=== DONE ===")
