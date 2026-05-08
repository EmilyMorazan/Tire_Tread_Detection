# Tire Tread Depth Detection — Demo Setup

## Requirements
- Node.js (v18 or newer) — download at https://nodejs.org

## Setup (one time)

```bash
# 1. Create a new React + Vite project
npm create vite@latest tire-tread-demo -- --template react
cd tire-tread-demo

# 2. Install dependencies
npm install
npm install recharts

# 3. Replace the default App component
# Copy TireTreadDemo.jsx into src/
# Then open src/main.jsx and make sure it says:
#   import App from './TireTreadDemo.jsx'

# 4. Start the dev server
npm run dev
```

Then open http://localhost:5173 in your browser.

## What's included in the demo

| Tab | What it shows |
|-----|--------------|
| **Predictor** | Sliders for tread depth, lighting, and camera angle → simulated model output with confidence score and safety status |
| **Upload** | Drag & drop a real tire photo → simulated prediction (swap for real model later) |
| **Distribution** | Class distribution bar chart — all 660 images across 9 classes |
| **Errors** | Per-class MAE chart with overall MAE reference line |
| **Training** | Full 155-epoch training curve across all 3 phases |

## Connecting your real model (optional, later)

When you get the `.pth` file from your teammates:

1. Set up a FastAPI backend:
```bash
pip install fastapi uvicorn torch torchvision pillow python-multipart
```

2. Create `backend.py`:
```python
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch, torchvision.transforms as T
from torchvision.models import resnet18
from PIL import Image
import io

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = resnet18()
model.fc = torch.nn.Linear(512, 1)
model.load_state_dict(torch.load("your_model.pth", map_location="cpu"))
model.eval()

transform = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    with torch.no_grad():
        pred = model(transform(img).unsqueeze(0)).item()
    return {"prediction": round(pred, 2)}
```

3. Run it: `uvicorn backend:app --reload`

4. In `TireTreadDemo.jsx`, replace the `simulatePredict` call in `UploadTab` with a real fetch to `http://localhost:8000/predict`.
