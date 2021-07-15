FROM nvcr.io/nvidia/pytorch:20.12-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget axel libopencv-dev build-essential cmake git curl \
        ca-certificates libjpeg-dev libpng-dev zip unzip megatools \
        python3 python3-pip python3-dev libpython3-dev

RUN apt-get install locales -y
RUN sed -i -e 's/# en_US.UTF-8/en_US.UTF-8/' /etc/locale.gen && locale-gen
ENV LANG=en_US.UTF-8 \
        LANGUAGE=en_US:en \
        LC_ALL=en_US.UTF-8

RUN pip3 install --upgrade pip setuptools

WORKDIR /retratista

COPY . /retratista

RUN pip3 install -r requirements.txt

RUN ./scripts/download_weights.sh

EXPOSE 5000

CMD python3 run_generator.py production 5000
