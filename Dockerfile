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
#COPY data data
COPY requirements.txt .

RUN mkdir output   && \
    mkdir data && \
    python -m pip install --upgrade pip  && \
    pip install -r requirements.txt && \
    gdown --folder https://drive.google.com/drive/folders/1D5wxqjiL1OiVL7EZXLY9YvkAXqlTtj2d?usp=sharing && \
#    gdown -O data/validation.tsv https://drive.google.com/file/d/1AIejLr5_mawJafnEqUPMKHp7xHziSy8J/view?usp=sharing && \
#    gdown -O data/train.tsv https://drive.google.com/file/d/18Pn9W_wV0FRard5yKsB_m2aq8sUn8AOS/view?usp=sharing && \
#    gdown -O data/test.tsv https://drive.google.com/file/d/12r1MetQa9Iwaw1ICfpFoce6bnqIji4Wd/view?usp=sharing && \
#    gdown -O data/text_prepare_tests.tsv https://drive.google.com/file/d/1aqDm7-opVMabgjxlyIuB32NCDi2D3bXy/view?usp=sharing && \
    python -u so_classifier/train_model.py


#EXPOSE 5000
#
#ENTRYPOINT [ "python" ]

#CMD [ "so_classifier/serve.py" ]
CMD gunicorn so_classifier.serve:app --bind 0.0.0.0:5000 --reload
