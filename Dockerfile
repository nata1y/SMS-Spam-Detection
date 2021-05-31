FROM python:3.8.10-slim

RUN apt-get update \
&& apt-get install -y --no-install-recommends git curl ffmpeg libsm6 libxext6 \
&& apt-get purge -y --auto-remove \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /root/

ENV VIRTUAL_ENV=/root/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONPATH="${PYTHONPATH}:/root/project/"

COPY requirements.txt .
RUN python -m pip install --upgrade pip &&\
    pip install cython &&\
    pip install -r requirements.txt &&\
    pip install alibi-detect

EXPOSE 8080

CMD "bash"
