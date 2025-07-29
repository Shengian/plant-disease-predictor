from flask import Flask, request, jsonify
import torch
from torchvision import transforms, models
from PIL import Image
import json
import os
from class_solution_generated import class_solutions

app = Flask(__name__)

# Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 138)  # 138 classes
model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device("cpu")))
model.eval()

# Load index to class mapping
with open("idx_to_class.json", "r") as f:
    idx_to_class = json.load(f)

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    try:
        img = Image.open(image).convert("RGB")
    except:
        return jsonify({"error": "Invalid image"}), 400

    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = str(predicted.item())
        class_name = idx_to_class[class_idx]
        solutions = class_solutions.get(class_name, ["No solution available"])

    return jsonify({
        "predicted_class": class_name,
        "solutions": solutions
    })

@app.route("/", methods=["GET"])
def home():
    return "Plant Disease Prediction API - 138 Class Version"

if __name__ == "__main__":
    app.run(debug=True)