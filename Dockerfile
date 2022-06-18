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


CMD gunicorn so_classifier.serve:app --bind 0.0.0.0:5000 --reload
