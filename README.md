# Vietnamese Legal Document Retrieval

Hệ thống truy xuất tài liệu pháp luật Việt Nam sử dụng kết hợp phương pháp Sparse (BM25) và Dense Retrieval (BGE-M3).

## Cấu trúc thư mục

```
VLSP2025-Vietnamese-Legal-Document-Retrieval/
├── data/
│   ├── raw/                    # Dữ liệu gốc
│   ├── processed/              # Dữ liệu đã xử lý
│   ├── faiss_index/            # FAISS index và metadata
│   ├── public_test/            # Dữ liệu test công khai
│   └── private_test/           # Dữ liệu test riêng tư
├── retrieve/
│   ├── dense/                  # Dense retrieval (BGE-M3)
│   └── sparse/                 # Sparse retrieval (BM25)
├── utils/                      # Các công cụ tiện ích
└── results/                    # Kết quả retrieve
```

## Yêu cầu hệ thống

```bash
pip install -r requirements.txt
```

Các thư viện chính:
- `faiss-cpu` hoặc `faiss-gpu`: Vector similarity search
- `transformers`: BGE-M3 model
- `rank_bm25`: BM25 algorithm
- `torch`: Deep learning framework

## Quy trình thực hiện

### Bước 1: Tạo Corpus

Chạy script để tạo corpus từ dữ liệu pháp luật gốc:

```bash
python utils/create_corpus.py
```

**Input:** `data/raw/legal_corpus.json`

**Output:** `data/processed/corpus.json`

Script này trích xuất các điều luật từ dữ liệu gốc và tạo corpus với định dạng:
```json
[
    {
        "aid": "article_id",
        "content_Article": "nội dung điều luật"
    }
]
```

### Bước 2: Chia tập dữ liệu

Tách dữ liệu train thành tập train và test để đánh giá:

```bash
python utils/split_data.py
```

**Input:** `data/raw/train.json`

**Output:**
- `data/processed/train.json` (80% dữ liệu)
- `data/processed/test.json` (20% dữ liệu)

Tỷ lệ chia: 80/20 với `random_seed=42` để đảm bảo tính tái tạo.

### Bước 3: Chunk Corpus

Chia corpus thành các đoạn ngắn hơn để cải thiện độ chính xác:

```bash
python utils/chunk.py
```

**Input:** `data/processed/corpus.json`

**Output:** `data/processed/chunked/chunk_corpus.json`

Mỗi điều luật được chia thành các chunks với:
- Độ dài tối đa mỗi chunk
- Overlap giữa các chunks để giữ ngữ cảnh
- Chunk ID theo format: `{article_id}_chunk_{index}`

### Bước 4: Sparse Retrieval (BM25)

#### 4.1. Tạo model BM25

```bash
cd retrieve/sparse
python create_model_bm25.py
```

Tạo và lưu model BM25 từ chunk corpus.

#### 4.2. Tìm kiếm với BM25

```bash
python search.py
```

**Output:** 
- `results/test/bm25_512_test.json`
- `results/private_test/bm25_512_private_test.json`

Định dạng kết quả:
```json
[
    {
        "qid": query_id,
        "top_chunks": [
            {"chunk_id": "article_id_chunk_0", "score": 15.23},
            ...
        ]
    }
]
```

### Bước 5: Dense Retrieval (BGE-M3)

#### 5.1. Tạo FAISS index

```bash
cd retrieve/dense
python create_corpus_meta.py
```

**Output:**
- `data/faiss_index/bge.bin` - FAISS index
- `data/faiss_index/corpus_meta.pkl` - Metadata mapping

#### 5.2. Predict với BGE-M3

```bash
python predict_bge.py
```

**Output:**
- `results/test/bge_512_test.json`
- `results/private_test/bge_512_private_test.json`

### Bước 6: Ensemble và Đánh giá

#### 6.1. Ensemble BM25 và BGE-M3

Kết hợp kết quả từ cả hai phương pháp:

```bash
python utils/ensemble_with_bm25.py
```

Các phương pháp ensemble:
- **sum**: MinMax normalization + weighted sum
- **product**: Raw score multiplication
- **product_rank**: Product + rank factor (recommended)

**Cấu hình trong script:**
```python
ENSEMBLE_METHOD = 'product_rank'  # Chọn phương pháp
custom_pairs = [
    {
        'model': 'bge_512_private_test.json',
        'model_weight': 1.0,
        'bm25': 'bm25_512_private_test.json',
        'bm25_weight': 1.0,
        'output': 'ensemble_bge_512_bm25_private_test.json',
    }
]
```

**Output:** `results/private_test/product_rank_ensemble_bge_512_bm25_private_test.json`

#### 6.2. Đánh giá trên tập test

```bash
python utils/evaluate.py
```

**Metrics:** Macro F2-score

**Cấu hình trong script:**
```python
TOPK = 3  # Số tài liệu top để đánh giá
PRED_PATHS = {
    "bge_model": ROOT / "results" / "test" / "bge_512_test.json",
    "ensemble": ROOT / "results" / "test" / "product_rank_ensemble_bge_512_bm25_test.json",
    # ... thêm các model khác
}
```

**Output:**
```
Loaded ground-truth for XXX queries from data/processed/test.json
Using TOPK = 3

bge_model: 0.7523
ensemble: 0.7891
...
```

## Tiện ích bổ sung

### Sắp xếp kết quả theo QID

```bash
python utils/sort_qid.py
```

Sắp xếp các entries trong file kết quả theo thứ tự QID tăng dần.

## Notes

1. **Chunk size**: Kích thước chunk ảnh hưởng đến độ chính xác. Thử nghiệm với các giá trị khác nhau (256, 512, 1024 tokens).

2. **Ensemble weights**: Điều chỉnh `model_weight` và `bm25_weight` để tối ưu kết quả:
   - BM25 tốt cho exact matching
   - BGE-M3 tốt cho semantic search
   - Ensemble thường cho kết quả tốt nhất

3. **TOPK**: Trong evaluate.py, TOPK=3 có nghĩa là chỉ xét 3 tài liệu top có điểm cao nhất.

4. **Reproducibility**: Sử dụng `random_seed=42` trong split_data.py để đảm bảo kết quả có thể tái tạo.

## License

See [LICENSE](LICENSE) file for details.

