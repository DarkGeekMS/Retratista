FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel

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

RUN pip install --upgrade pip setuptools

WORKDIR /retratista

COPY . /retratista

RUN pip install Flask Flask-CORS numpy scipy matplotlib opencv-python \
                scikit-image dominate dill face-alignment==1.0.0 cython tensorboard future

RUN pip install neural_renderer_pytorch

RUN cd ./src/pose_lib/utils/cython && python setup.py build_ext -i && cd ../../../.. 

RUN ./scripts/download_weights_rot.sh

EXPOSE 5001

CMD python3 run_rotator.py production 5001
