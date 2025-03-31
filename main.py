from flask import Flask, request, jsonify
from PIL import Image
import requests
import io
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import base64
import struct

app = Flask(__name__)

# Load a lightweight pretrained model (e.g. MobileNetV2)
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Resize + normalize image like the model expects
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def encode_vector_base64(vector):
    float_array = vector.detach().cpu().numpy().astype("float32")
    byte_array = struct.pack(f'{len(float_array)}f', *float_array)
    return base64.urlsafe_b64encode(byte_array).decode("utf-8")

@app.route("/")
def index():
    return "DeepVision API is running"

@app.route("/embed")
def embed_image():
    url = request.args.get("url")
    if not url:
        return jsonify({"error": "Missing image URL"}), 400

    try:
        response = requests.get(url)
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        input_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            features = model(input_tensor).squeeze()

        encoded = encode_vector_base64(features)
        return jsonify({"embedding": encoded})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
