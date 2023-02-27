from pydantic import BaseModel
from fastapi import File, UploadFile
from typing import Literal

class ModelRequest(BaseModel):
    model_size: Literal[
        "base",
        "small", 
        "medium", 
        "large", 
        "large-v2"
    ]
    model_language: Literal[
        "en",
        "de",
        "es",
        "fr",
    ]
    
    
class LanguageRequest(BaseModel):
    model_language: Literal[
        "en",
        "de",
        "es",
        "fr",
    ]
    
    
class AudioFeatures(BaseModel):
    file: UploadFile = File(...)
    
class SpeakerRequest(BaseModel):
    num_speakers: int
    