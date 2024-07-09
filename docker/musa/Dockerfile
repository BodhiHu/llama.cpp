FROM ubuntu:20.04

RUN sed -i "s/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g" /etc/apt/sources.list
RUN apt-get update && \
    apt-get install -y build-essential python3 python3-pip git libcurl4-openssl-dev vim

COPY requirements.txt   requirements.txt
COPY requirements       requirements

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install --upgrade pip setuptools wheel \
#     && pip install -r requirements.txt

WORKDIR /app

# Enable MUSA
ENV LLAMA_MUSA=1
# Enable cURL
ENV LLAMA_CURL=1

# COPY . .

CMD sleep infinity
