import json
from torchvision import datasets

# === Set path to your dataset folder ===
data_dir = "ml training/new linked"

# === Use ImageFolder to get class-to-index mapping ===
dataset = datasets.ImageFolder(data_dir)
class_to_idx = dataset.class_to_idx

# === Flip it to index-to-class mapping ===
idx_to_class = {v: k for k, v in class_to_idx.items()}

# === Save to JSON file ===
with open("idx_to_class.json", "w") as f:
    json.dump(idx_to_class, f, indent=4)

print("✅ Saved index-to-class mapping to idx_to_class.json")