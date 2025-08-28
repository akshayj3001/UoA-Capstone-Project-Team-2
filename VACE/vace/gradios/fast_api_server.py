import nest_asyncio
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

nest_asyncio.apply()

app = FastAPI()

class RequestModel(BaseModel):
    transcript: str
    audio_features: Dict[str, Any]

@app.post("/gpt-oss")
def generate(request: RequestModel):
  from transcript_to_video_prompt import generate_video_prompt

  prompt = generate_video_prompt(request.transcript, request.audio_features)
  return prompt


uvicorn.run(app, host="127.0.0.1", port=8000)
