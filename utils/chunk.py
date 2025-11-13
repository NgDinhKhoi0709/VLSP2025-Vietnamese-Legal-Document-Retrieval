import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

input_path = "./data/processed/corpus.json"
output_path = "./data/processed/chunked/chunk_corpus.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,              # 400 từ
    chunk_overlap=50,            # overlap 50 từ
    length_function=lambda s: len(s.split()),
    separators=["\n\n", "\n", " ", ""],  # giữ logic cắt gọn gàng
)

chunked_data = []

for item in data:
    aid = item["aid"]
    content = item["content_Article"]
    chunks = text_splitter.split_text(content)
    for idx, chunk in enumerate(chunks):
        chunked_data.append({
            "aid": aid,
            "chunk_id": f"{aid}_{idx}",
            "content_Article": chunk
        })

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(chunked_data, f, ensure_ascii=False, indent=4)