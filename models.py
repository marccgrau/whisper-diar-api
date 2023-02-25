import whisper
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from utils import utils
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import librosa
import io
import tempfile

class ASRDiarModel:
    def __init__(self, device):
        self.BASE_ASRTRUNC = "openai/whisper-"
        self.BASE_EMBEDDINGMODEL = "speechbrain/spkrec-ecapa-voxceleb"
        self.BASE_LANGUAGE = "en"
        self.BASE_NUM_SPEAKERS = 2
        self.device = device
        self.model_size = "base"
        self.asrmodel = whisper.load_model(self.model_size, device=self.device)
        self.embeddingmodel = PretrainedSpeakerEmbedding(self.BASE_EMBEDDINGMODEL, device=self.device)
        self.model_language = self.BASE_LANGUAGE
        self.num_speakers = self.BASE_NUM_SPEAKERS
        
    def load_new_asrmodel(self, model_size, model_language):
        self.model_size = model_size
        self.model_language = model_language
        self.asrmodel = whisper.load_model(self.model_size, device=self.device)
        #self.asrmodel.config.forced_decoder_ids = self.asrmodel.tokenizer.get_decoder_prompt_ids(language=self.model_language, task="transcribe")
    
    def change_language(self, model_language):
        self.model_language = model_language
        #self.asrmodel.config.forced_decoder_ids = self.asrmodel.tokenizer.get_decoder_prompt_ids(language=self.model_language, task="transcribe")
        
    def change_num_speakers(self, num_speakers):
        self.num_speakers = num_speakers
    
    def transcribe(self, audio_file):
        audio_file.seek(0)
        spooled_data = audio_file.file.read()
        named_file = tempfile.NamedTemporaryFile(delete=False)
        named_file.write(spooled_data)
        audio_file.close()
        audio = utils.convert_bytes_to_wav(spooled_data)
        del spooled_data
        duration = librosa.get_duration(audio)
        options = dict(language=self.model_language, beam_size=3, best_of=3)
        transcribe_options = dict(task="transcribe", **options)
        result = self.asrmodel.transcribe(audio, **transcribe_options)
        segments = result["segments"]
        
        # get embedding for each segment
        embeddings = np.zeros(shape=(len(segments), 192))
        
        for i, segment in enumerate(segments):
            embeddings[i] = utils.segment_embedding(named_file, segment, duration, self)
        embeddings = np.nan_to_num(embeddings)
        
        named_file.close()
        # assign speaker labels to each segment        
        if self.num_speakers == 1:
            for i in range(len(segments)):
                segments[i]["speaker"] = 'Speaker 1'
        else:
            clustering = AgglomerativeClustering(self.num_speakers).fit(embeddings)
            labels = clustering.labels_
            for i in range(len(segments)):
                segments[i]["speaker"] = f'Speaker {labels[i] + 1}'
        
        objects = {
            'Start': [],
            'End': [],
            'Speaker': [],
            'Text': [],
        }
        
        text = ''
        
        if self.num_speakers == 1:
            objects['Start'].append(str(utils.convert_time(segment['start'])))
            objects['Speaker'].append(segment['speaker'])
            for (i, segment) in enumerate(segments):
                text += segment['text'] + ' '
            objects['Text'].append(text)
            objects['End'].append(str(utils.convert_time(segment['end'])))
        else:
            for (i, segment) in enumerate(segments):
                if i == 0 or segments[i-1]['speaker'] != segment['speaker']:
                    objects['Start'].append(str(utils.convert_time(segment['start'])))
                    objects['Speaker'].append(segment['speaker'])
                    if i != 0:
                        objects['End'].append(str(utils.convert_time(segments[i-1]['end'])))
                        objects['Text'].append(text)
                        text = ''
                text += segment['text'] + ' '
            objects['End'].append(str(utils.convert_time(segments[i-1]['end'])))
            objects['Text'].append(text)
        
        return objects
    
    