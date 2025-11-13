"""
Script to create corpus_meta.pkl from FAISS index and chunk corpus
"""
import json
import pickle
import faiss
import argparse


def create_corpus_meta(path_index: str, path_chunk: str, out_meta: str):
    """
    Tạo file corpus_meta.pkl từ FAISS index và chunk corpus
    
    Args:
        path_index: Đường dẫn đến FAISS index
        path_chunk: Đường dẫn đến chunk corpus JSON
        out_meta: Đường dẫn output để lưu corpus_meta.pkl
    """
    # 1) Load FAISS index để biết total vectors
    print("Loading FAISS index...")
    index = faiss.read_index(path_index)
    N = index.ntotal
    print(f"Index vectors: {N}")

    # 2) Load corpus (đảm bảo đúng thứ tự khi build index)
    print("Loading chunk corpus...")
    with open(path_chunk, "r", encoding="utf-8") as f:
        chunk_data = json.load(f)

    # 3) Build meta
    print("Building metadata...")
    meta = []
    for i, item in enumerate(chunk_data):
        # Bắt buộc phải có hai trường này
        if "aid" not in item or "chunk_id" not in item:
            raise ValueError(f"Item thứ {i} thiếu 'aid' hoặc 'chunk_id'")
        aid = item["aid"]
        cid = item["chunk_id"]
        meta.append((aid, cid))

    print(f"Meta entries: {len(meta)}")

    # 4) Kiểm tra khớp số lượng
    if len(meta) != N:
        raise ValueError(
            f"Meta length ({len(meta)}) khác index.ntotal ({N}). "
            "Hãy kiểm tra lại thứ tự hoặc số chunk trong corpus."
        )

    # 5) Lưu corpus_meta.pkl
    print(f"Saving metadata to {out_meta}...")
    with open(out_meta, "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Đã lưu meta vào {out_meta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create corpus_meta.pkl from FAISS index and chunk corpus"
    )
    parser.add_argument(
        "--path_index",
        type=str,
        required=True,
        help="Path to FAISS index file"
    )
    parser.add_argument(
        "--path_chunk",
        type=str,
        required=True,
        help="Path to chunk corpus JSON file"
    )
    parser.add_argument(
        "--out_meta",
        type=str,
        required=True,
        help="Output path for corpus_meta.pkl"
    )
    
    args = parser.parse_args()
    
    create_corpus_meta(
        path_index=args.path_index,
        path_chunk=args.path_chunk,
        out_meta=args.out_meta
    )

