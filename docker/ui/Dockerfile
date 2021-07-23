FROM vuejs/ci:latest

ARG TYPE

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends megatools

RUN apt-get install locales -y
RUN sed -i -e 's/# en_US.UTF-8/en_US.UTF-8/' /etc/locale.gen && locale-gen
ENV LANG=en_US.UTF-8 \
        LANGUAGE=en_US:en \
        LC_ALL=en_US.UTF-8

WORKDIR /retratista

COPY ./ui /retratista

RUN ./download_demo.sh

RUN npm install
RUN npm install -S vue-sweetalert2
RUN npm i aos
RUN npm install autoprefixer@9.8.6
RUN npm i particle-bg
RUN npm install --save axios vue-axios
RUN npm rebuild node-sass

EXPOSE 8080

CMD npm run $TYPE
