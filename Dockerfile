FROM python:3.10-slim

RUN apt-get update \
&& apt-get install -y --no-install-recommends git \
&& apt-get purge -y --auto-remove \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /root/

ENV VIRTUAL_ENV=/root/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY test test
COPY so_classifier so_classifier
COPY requirements.txt .

RUN mkdir model   && \
    mkdir data && \
    python -m pip install --upgrade pip  && \
    pip install -r requirements.txt && \
    gdown --folder https://drive.google.com/drive/folders/1InFeBLhOU-Y2Sj8mjE-2IKFs9H5rby3K?usp=sharing
#    gdown --folder https://drive.google.com/drive/folders/1D5wxqjiL1OiVL7EZXLY9YvkAXqlTtj2d?usp=sharing && \
#    gdown -O data/online.tsv https://docs.google.com/spreadsheets/d/1XeQkfdNCQB8L1EmwSEzgMeOSq3bXoKBh9JN337UGhSI/export?format=tsv && \
#    python -u so_classifier/train_model.py


CMD gunicorn so_classifier.serve:app --bind 0.0.0.0:5000 --reload
