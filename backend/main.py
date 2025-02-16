import torch
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models.model import SentimentModel

# Load the vocabulary and initialize the model
vocab = np.load("./datasets/vocab.npy", allow_pickle=True).item()
model = SentimentModel(len(vocab), embedding_dim=100, hidden_dim=128, threshold=0.8)

# GPU for computations
device = torch.device("cuda")

# Load the saved weights and set it to evaluation mode
model.load_state_dict(torch.load("./models/model_weights.pt"))
model.to(device)
model.eval()

# Initialize FastAPI server and add the SvelteKit address
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


class TextInput(BaseModel):
    text: str


@app.post("/analyze/")
async def analyze_text(data: TextInput):
    text = data.text
    prediction = model.predict(text, vocab)
    output = f"{prediction[0]} with confidence {prediction[1]:.2f}"
    return output
