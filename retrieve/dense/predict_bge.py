"""
Script to perform dense retrieval using BGE M3 model and FAISS index
"""
import json
import torch
import faiss
import pickle
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer


def load_data(path_test: str, path_index: str, path_meta: str):
    """
    Load test queries, FAISS index, and metadata
    
    Args:
        path_test: Path to test queries JSON
        path_index: Path to FAISS index
        path_meta: Path to corpus metadata pickle file
        
    Returns:
        queries: List of query dictionaries
        index: FAISS index object
        meta: List of (aid, chunk_id) tuples
    """
    print("Loading test queries...")
    with open(path_test, "r", encoding="utf-8") as f:
        queries = json.load(f)
    
    print("Loading FAISS index...")
    index = faiss.read_index(path_index)
    
    print("Loading metadata...")
    with open(path_meta, "rb") as f:
        meta = pickle.load(f)
    
    return queries, index, meta


def load_model(model_path: str):
    """
    Load BGE M3 model
    
    Args:
        model_path: Path to the model checkpoint
        
    Returns:
        model: SentenceTransformer model
        device: Device (cuda or cpu)
    """
    print("Loading BGE M3 model...")
    model = SentenceTransformer(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on device: {device}")
    
    return model, device


def search_and_build_results(queries, model, index, meta, topk: int = 100):
    """
    Encode queries, search FAISS index, and build results
    
    Args:
        queries: List of query dictionaries with 'qid' and 'question'
        model: SentenceTransformer model
        index: FAISS index
        meta: Metadata list of (aid, chunk_id) tuples
        topk: Number of top results to retrieve
        
    Returns:
        output: List of results with qid and top_chunks
    """
    print(f"Processing {len(queries)} queries...")
    output = []
    
    for i, q in enumerate(queries):
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(queries)} queries...")
            
        qid = q["qid"]
        question = q["question"]

        # Encode query
        q_emb = model.encode(
            question, 
            normalize_embeddings=True, 
            convert_to_numpy=True
        )

        # Search top-k
        D, I = index.search(np.array([q_emb]), k=topk)

        # Chuyển thành list các dict {\"chunk_id\": ..., \"score\": ...}
        top_chunks = [
            {"chunk_id": meta[idx][1], "score": float(D[0][j])}
            for j, idx in enumerate(I[0])
        ]

        output.append({
            "qid": qid,
            "top_chunks": top_chunks
        })
    
    return output


def save_results(output, output_file: str):
    """
    Save results to JSON file
    
    Args:
        output: Results to save
        output_file: Output file path
    """
    print(f"Saving results to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print("✅ Đã lưu kết quả")


def main():
    """Main function to run the prediction pipeline"""
    parser = argparse.ArgumentParser(
        description="Dense retrieval using BGE M3 model and FAISS index"
    )
    parser.add_argument(
        "--path_test",
        type=str,
        required=True,
        help="Path to test queries JSON file"
    )
    parser.add_argument(
        "--path_index",
        type=str,
        required=True,
        help="Path to FAISS index file"
    )
    parser.add_argument(
        "--path_meta",
        type=str,
        required=True,
        help="Path to corpus_meta.pkl file"
    )
    parser.add_argument(
        "--path_model",
        type=str,
        required=True,
        help="Path to BGE M3 model checkpoint"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file path for results JSON"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="Number of top results to retrieve (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Load data
    queries, index, meta = load_data(args.path_test, args.path_index, args.path_meta)
    
    # Load model
    model, device = load_model(args.path_model)
    
    # Search and build results
    output = search_and_build_results(queries, model, index, meta, topk=args.topk)
    
    # Save results
    save_results(output, args.output_file)


if __name__ == "__main__":
    main()

