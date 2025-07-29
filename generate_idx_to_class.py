import os
import json

# Use raw string to avoid path issues with spaces
data_dir = r"D:/ml training/new linked"

# Get all folder names (classes), sorted for consistency
class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

# Create index to class mapping
idx_to_class = {idx: class_name for idx, class_name in enumerate(class_names)}

# Save to JSON
with open("idx_to_class.json", "w") as f:
    json.dump(idx_to_class, f, indent=4)

print(f"Mapping saved with {len(idx_to_class)} classes to idx_to_class.json")