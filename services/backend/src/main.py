import uvicorn
from logging.config import dictConfig
import logging
from utils import config
from services.backend.src.schemas import ModelRequest, LanguageRequest, SpeakerRequest
from fastapi.middleware.cors import CORSMiddleware

import torch
from fastapi import FastAPI, File, UploadFile
from services.backend.src.models import ASRDiarModel


dictConfig(config.LogConfig().dict())
logger = logging.getLogger("asr-diarization")


BASE_MODEL = "openai/whisper-"
BASE_LANG = "en"

device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = ASRDiarModel(device)

@app.on_event("startup")
async def startup_event():
    logger.info(f"Device: {model.device}")
    logger.info(f"Loaded default ASR pipeline with default language {BASE_LANG} on {device}")
    logger.info(f"Loaded speaker embedding model on {model.device}")
    logger.info(f"Default number of speakers: {model.num_speakers}")
    return {"message": "ASR-Diarization API started"}
    
@app.get("/")
async def welcome_message():
    return {"message": "Welcome to the ASR-Diarization API"}

@app.get("/overview")
async def overview():
    message = {
        "ASR-Model": model.BASE_ASRTRUNC + model.model_size,
        "Embedding-Model": model.BASE_EMBEDDINGMODEL,
        "Language": model.model_language,
        "Number of speakers": model.num_speakers,
    }
    return message


@app.post("/load_new_model")
async def load_setting(model_request: ModelRequest):
    model_size = model_request.model_size
    model_language = model_request.model_language
    model.load_new_asrmodel(model_size, model_language)
    logger.info(f"Loaded new ASR model {model.BASE_ASRTRUNC + model.model_size} on {model.device} with language {model.model_language}")
    return {"message": "Changed ASR-model and diarization model to new setting"}

    
@app.post("/change_language")
async def change_language(language_request: LanguageRequest):
    model_language = language_request.model_language
    model.change_language(model_language)
    logger.info(f"Changed language to {model.model_language}")
    return {"message": f"Changed language to {model.model_language}"}

@app.post("/change_speakers")
async def change_speakers(speaker_request: SpeakerRequest):
    num_speakers = speaker_request.num_speakers
    model.change_num_speakers(num_speakers)
    logger.info(f"Changed number of speakers to {model.num_speakers}")
    return {"message": f"Changed number of speakers to {model.num_speakers}"}
    

@app.post("/transcribe_embed")
async def transcribe(file: UploadFile = File(...)):
    return model.transcribe(file)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)