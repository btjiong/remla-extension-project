FROM python:3.10-slim

RUN apt-get update \
&& apt-get install -y --no-install-recommends git \
&& apt-get purge -y --auto-remove \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /root/

ENV VIRTUAL_ENV=/root/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .
RUN python -m pip install --upgrade pip  &&\
    pip install -r requirements.txt

COPY so_classifier so_classifier

COPY data data

EXPOSE 8080

ENTRYPOINT [ "python3" ]

CMD [ "so_classifier/serve.py" ]