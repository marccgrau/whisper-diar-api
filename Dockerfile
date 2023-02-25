FROM python:3.10-slim

COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install git -y
RUN apt-get install build-essential -y
RUN pip3 install -r requirements.txt
RUN pip3 install git+https://github.com/openai/whisper.git

EXPOSE 8000

CMD ["python3", "main.py"]