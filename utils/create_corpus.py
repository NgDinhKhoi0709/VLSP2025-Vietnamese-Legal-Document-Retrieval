import json

input_path = "./data/raw/legal_corpus.json"
output_path = "./data/processed/corpus.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

filtered_articles = []
for law in data:
    for article in law["content"]:
        filtered_articles.append({
            "aid": article["aid"],
            "content_Article": article["content_Article"]
        })

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(filtered_articles, f, ensure_ascii=False, indent=4)
