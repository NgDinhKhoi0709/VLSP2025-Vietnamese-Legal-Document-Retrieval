import json

# Đường dẫn file
file_private_test = "data/private_test/private_test.json"
file_results = "results/private_test/product_rank_ensemble_bge_512_bm25_private_test.json"
file_output = "results/private_test/product_rank_ensemble_bge_512_bm25_private_test_sorted.json"

# 1. Đọc private_test.json để lấy thứ tự qid
with open(file_private_test, "r", encoding="utf-8") as f:
    private_test_data = json.load(f)

qid_order = [item["qid"] for item in private_test_data]

# 2. Đọc file kết quả
with open(file_results, "r", encoding="utf-8") as f:
    results_data = json.load(f)

# 3. Tạo dictionary qid → dữ liệu để tra cứu nhanh
results_dict = {item["qid"]: item for item in results_data}

# 4. Sắp xếp lại theo thứ tự qid của private_test.json
sorted_results = [results_dict[qid] for qid in qid_order if qid in results_dict]

# 5. Ghi ra file mới
with open(file_output, "w", encoding="utf-8") as f:
    json.dump(sorted_results, f, ensure_ascii=False, indent=2)

print(f"✅ Đã sắp xếp xong và lưu vào: {file_output}")

from collections import defaultdict

# Đọc file abc.json
with open("results/private_test/product_rank_ensemble_bge_512_bm25_private_test_sorted.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Tạo dict để đếm chunk_id cho mỗi qid
counts = defaultdict(int)

for item in data:
    qid = item["qid"]
    counts[qid] = len(item.get("top_chunks", []))

# In kết quả
for qid, count in counts.items():
    print(f"qid {qid} có {count} chunk_id")
