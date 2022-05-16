FROM python:3.7.10-slim

RUN apt-get update \
&& apt-get install -y --no-install-recommends git \
&& apt-get purge -y --auto-remove \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /root/

ENV VIRTUAL_ENV=/root/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip3 install jupyter

COPY stack.ipynb .
ADD data ./data

RUN jupyter nbconvert --to script stack.ipynb
#RUN ipython stack.py
