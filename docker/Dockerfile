FROM tensorflow/tensorflow:2.6.1-gpu-jupyter

RUN apt update
RUN apt install -y git libsm6 libxrender1 libgl1-mesa-glx wget zip unzip

RUN pip install --upgrade setuptools wheel

ADD . /workspace/parnet-tf2/docker/
WORKDIR /workspace/parnet-tf2/docker

RUN pip install -r requirements.txt
WORKDIR /workspace