import json
import pickle
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize
import string
import os
from tqdm import tqdm

# Stopword giống corpus
number = [str(i) for i in range(1, 11)]
chars = list("abcdefghijklmnoprstuvxyđ")
stop_word = number + chars + [
    "của", "và", "các", "có", "được", "theo", "tại", "trong", "về", 
    "hoặc", "người",  "này", "khoản", "cho", "không", "từ", "phải", 
    "ngày", "việc", "sau",  "để",  "đến", "bộ",  "với", "là", "năm", 
    "khi", "số", "trên", "khác", "đã", "thì", "thuộc", "điểm", "đồng",
    "do", "một", "bị", "vào", "lại", "ở", "nếu", "làm", "đây", 
    "như", "đó", "mà", "nơi", "”", "“"
]

def remove_stopword(w):
    return w not in stop_word

def remove_punctuation(w):
    return w not in string.punctuation

def lower_case(w):
    return w.lower()

def bm25_tokenizer(text):
    tokens = word_tokenize(text)
    tokens = list(map(lower_case, tokens))
    tokens = list(filter(remove_punctuation, tokens))
    tokens = list(filter(remove_stopword, tokens))
    return tokens

# Load câu hỏi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir, os.pardir))

# Construct paths relative to project root
TEST_PATH = os.path.join(ROOT_DIR, "data/private_test/private_test.json")
CHUNK_CORPUS_PATH = os.path.join(ROOT_DIR, "data/processed/chunked/chunk_corpus.json")

MODEL_PATH = os.path.join(BASE_DIR, "bm25_model.pkl")

OUTPUT_PATH = os.path.join(ROOT_DIR, "results", "private_test", "bm25_512_private_test.json")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Load câu hỏi
with open(TEST_PATH, "r", encoding="utf-8") as f:
    question_data = json.load(f)

# Load mô hình BM25
with open(MODEL_PATH, "rb") as f:
    bm25_model = pickle.load(f)

# Load chunk_id gốc (dùng để truy vết)
with open(CHUNK_CORPUS_PATH, "r", encoding="utf-8") as f:
    chunk_data = json.load(f)
chunk_ids = [item["chunk_id"] for item in chunk_data]

# Truy xuất top 100 chunk_id cho từng câu hỏi
results = []

for entry in tqdm(question_data, desc="Processing Questions"):
    qid = entry["qid"]
    question = entry["question"]

    tokenized_query = bm25_tokenizer(question)
    scores = bm25_model.get_scores(tokenized_query)

    top_n = 2000
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]

    top_chunks = [
        {"chunk_id": chunk_ids[i], "score": float(scores[i])}
        for i in top_indices
    ]

    results.append({
        "qid": qid,
        "question": question,
        "top_chunks": top_chunks
    })

# Lưu kết quả ra file JSON
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
