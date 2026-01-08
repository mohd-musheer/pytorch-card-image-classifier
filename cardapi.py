from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
import io
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super().__init__()
        self.base = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.base(x)

model = SimpleCardClassifier(53)

if os.path.exists("card_model.pth"):
    model.load_state_dict(torch.load("card_model.pth", map_location=device))

model.to(device)
model.eval()

target_to_class = [
    'ace of clubs','ace of diamonds','ace of hearts','ace of spades',
    'eight of clubs','eight of diamonds','eight of hearts','eight of spades',
    'five of clubs','five of diamonds','five of hearts','five of spades',
    'four of clubs','four of diamonds','four of hearts','four of spades',
    'jack of clubs','jack of diamonds','jack of hearts','jack of spades',
    'joker','king of clubs','king of diamonds','king of hearts','king of spades',
    'nine of clubs','nine of diamonds','nine of hearts','nine of spades',
    'queen of clubs','queen of diamonds','queen of hearts','queen of spades',
    'seven of clubs','seven of diamonds','seven of hearts','seven of spades',
    'six of clubs','six of diamonds','six of hearts','six of spades',
    'ten of clubs','ten of diamonds','ten of hearts','ten of spades',
    'three of clubs','three of diamonds','three of hearts','three of spades',
    'two of clubs','two of diamonds','two of hearts','two of spades'
]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return JSONResponse({
        "prediction": target_to_class[pred.item()].upper(),
        "confidence": f"{conf.item() * 100:.2f}%"
    })
