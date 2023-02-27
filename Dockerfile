FROM python:3.10-slim

WORKDIR /backend
ADD /backend/ /backend/

COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install git -y
RUN apt-get install build-essential -y
RUN apt-get install libsndfile1 -y
RUN pip3 install -r requirements.txt

EXPOSE 8000

CMD ["python3", "main.py"]