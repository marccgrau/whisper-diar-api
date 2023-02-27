import os
import contextlib
import wave
from pyannote.audio import Audio
from pyannote.core import Segment
import datetime
import io
import librosa

def convert_to_wav(filepath):
    _,file_ending = os.path.splitext(f'{filepath}')
    audio_file = filepath.replace(file_ending, ".wav")
    os.system(f'ffmpeg -i "{filepath}" -ar 16000 -ac 1 -c:a pcm_s16le "{audio_file}"')
    return audio_file

def convert_bytes_to_wav(data):
    audio = librosa.load(io.BytesIO(data), sr=16000)[0]
    return audio
    

def get_duration(filepath):
    with contextlib.closing(wave.open(filepath,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration
    
def segment_embedding(audio_file, segment, duration, model):
    audio = Audio()
    start = segment["start"]
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(audio_file.name, clip)
    return model.embeddingmodel(waveform[None])

def convert_time(secs):
    return datetime.timedelta(seconds=round(secs))
    