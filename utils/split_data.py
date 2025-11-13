import json
import random

input_path = "./data/raw/train.json"
train_path = "./data/processed/train.json"
test_path = "./data/processed/test.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

random.seed(42)  # Đảm bảo kết quả chia reproducible
random.shuffle(data)

split_index = int(0.8 * len(data))
train_data = data[:split_index]
test_data = data[split_index:]

with open(train_path, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open(test_path, "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)
