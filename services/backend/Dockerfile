FROM python:3.10-slim

RUN mkdir app
WORKDIR /app

ENV PATH="${PATH}:/root/.local/bin"
ENV PYTHONPATH=.

COPY requirements.txt .
RUN apt-get update && apt-get install git -y
RUN apt-get install build-essential -y
RUN apt-get install libsndfile1 -y
RUN pip3 install -r requirements.txt

COPY src/ .

#EXPOSE 8000

#CMD ["python3", "main.py"]