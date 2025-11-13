import json
import string
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize
import pickle
from tqdm import tqdm

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

with open("./data/processed/chunked/chunk_corpus.json", "r", encoding="utf-8") as f:
    chunk_data = json.load(f)

law_chunks = [item["content_Article"] for item in chunk_data]

tokenized_chunks = [bm25_tokenizer(text) for text in tqdm(law_chunks, desc="Tokenizing")]

bm25_model = BM25Okapi(tokenized_chunks)

with open("./retrieve/sparse/bm25_model.pkl", "wb") as f:
    pickle.dump(bm25_model, f)
