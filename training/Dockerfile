ARG cuda_version=10.0
ARG cudnn_version=7
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel

# ARG cuda_version=11.2.2
# ARG cudnn_version=8
# FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel

# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update \
  && apt-get install -y \
  python3-dev \
  python3-pip \
  libatlas-base-dev \
  libhdf5-dev \
  libsm6 \
  libxext6 \
  libxrender-dev \
  git \
  wget \
  && rm -rf /var/lib/apt/lists/*

RUN pip3 --no-cache-dir install --upgrade pip

COPY requirements.txt requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

ARG SAGE_USER_TOKEN="-10"
ARG SAGE_HOST="HOST"
ARG BUCKET_ID_MODEL="BUCKET_ID_MODEL"
ARG LC_ALL="C.UTF-8"
ARG LANG="C.UTF-8"
ENV LC_ALL="C.UTF-8" \
    LANG="C.UTF-8" \
    WAGGLE_PLUGIN_ID="50" \
    WAGGLE_PLUGIN_NAME="Smoke Detection Model" \
    WAGGLE_PLUGIN_VERSION="0.3.0" \
    WAGGLE_PLUGIN_INSTANCE="1" \
    WAGGLE_PLUGIN_REF="https://github.com/iperezx/edge-plugins/tree/master/plugin-smokedetect" \
    SAGE_USER_TOKEN=${SAGE_USER_TOKEN} \
    SAGE_HOST=${SAGE_HOST} \
    BUCKET_ID_MODEL=${BUCKET_ID_MODEL}

WORKDIR /userdata/kerasData
