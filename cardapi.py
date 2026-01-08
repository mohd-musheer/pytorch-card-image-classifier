from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
import io
import os

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

# Initialize and load model
NUM_CLASSES = 53
model = SimpleCardClassifier(NUM_CLASSES)

# Check if model exists before loading
MODEL_PATH = "card_model.pth"
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    print(f"Warning: {MODEL_PATH} not found. Running with uninitialized weights.")

model.to(device)
model.eval()

target_to_class = {
    0: 'ace of clubs', 1: 'ace of diamonds', 2: 'ace of hearts', 3: 'ace of spades',
    4: 'eight of clubs', 5: 'eight of diamonds', 6: 'eight of hearts', 7: 'eight of spades',
    8: 'five of clubs', 9: 'five of diamonds', 10: 'five of hearts', 11: 'five of spades',
    12: 'four of clubs', 13: 'four of diamonds', 14: 'four of hearts', 15: 'four of spades',
    16: 'jack of clubs', 17: 'jack of diamonds', 18: 'jack of hearts', 19: 'jack of spades',
    20: 'joker',
    21: 'king of clubs', 22: 'king of diamonds', 23: 'king of hearts', 24: 'king of spades',
    25: 'nine of clubs', 26: 'nine of diamonds', 27: 'nine of hearts', 28: 'nine of spades',
    29: 'queen of clubs', 30: 'queen of diamonds', 31: 'queen of hearts', 32: 'queen of spades',
    33: 'seven of clubs', 34: 'seven of diamonds', 35: 'seven of hearts', 36: 'seven of spades',
    37: 'six of clubs', 38: 'six of diamonds', 39: 'six of hearts', 40: 'six of spades',
    41: 'ten of clubs', 42: 'ten of diamonds', 43: 'ten of hearts', 44: 'ten of spades',
    45: 'three of clubs', 46: 'three of diamonds', 47: 'three of hearts', 48: 'three of spades',
    49: 'two of clubs', 50: 'two of diamonds', 51: 'two of hearts', 52: 'two of spades'
}

# Added Normalization (standard for EfficientNet)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_html_content():
    with open("index.html", "r") as f:
        return f.read()

@app.get("/", response_class=HTMLResponse)
async def index():
    # Show "Ready for upload" instead of the raw placeholder on first load
    return get_html_content().replace("{{prediction}}", "Waiting for image...")

@app.post("/", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocess and Predict
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
        
        prediction = target_to_class.get(pred.item(), "Unknown Card")
        
    except Exception as e:
        prediction = f"Error processing image: {str(e)}"

    return get_html_content().replace("{{prediction}}", prediction.upper())
