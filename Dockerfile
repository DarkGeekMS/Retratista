FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

ENV ANACONDA /opt/anaconda3
ENV CUDA_PATH /usr/local/cuda
ENV PATH ${ANACONDA}/bin:${CUDA_PATH}/bin:$PATH
ENV LD_LIBRARY_PATH ${ANACONDA}/lib:${CUDA_PATH}/bin64:$LD_LIBRARY_PATH
ENV C_INCLUDE_PATH ${CUDA_PATH}/include

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget axel libopencv-dev build-essential cmake git curl \
        ca-certificates libjpeg-dev libpng-dev zip unzip megatools

RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -P /tmp
RUN bash /tmp/Anaconda3-2020.11-Linux-x86_64.sh -b -p $ANACONDA
RUN rm -rf /tmp

RUN pip3 install --upgrade pip

WORKDIR /retratista

COPY . /retratista

RUN pip3 install -r requirements.txt

RUN ./scripts/download_weights.sh

EXPOSE 5000

CMD python3 run.py production 5000
