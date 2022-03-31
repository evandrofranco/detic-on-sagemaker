# Use nvidia/cuda image
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# install anaconda
RUN apt-get update
RUN apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
        apt-get clean

# Required for open CV
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         python3-pip \
         python3-setuptools \
         nginx \
         ca-certificates \
         make \
         automake \
         gcc \
         g++ \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy

# set path to conda
ENV PATH /opt/conda/bin:$PATH

RUN conda update conda \
    && conda create --name detic python=3.8

RUN echo "conda activate detic" >> ~/.bashrc
ENV PATH /opt/conda/envs/detic/bin:$PATH
ENV CONDA_DEFAULT_ENV detic

RUN pip --no-cache-dir install pandas flask gunicorn jsonpickle

# Install Torch
RUN pip install torch==1.9.0 torchvision==0.10.0
RUN conda install cudatoolkit=11.1 -c nvidia -y

# Install Detectron2
#RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html
RUN git clone https://github.com/facebookresearch/detectron2.git \
    && cd detectron2 \
    && pip install -e .

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Clone and install Detic
RUN git clone https://github.com/facebookresearch/Detic.git --recurse-submodules /opt/program
WORKDIR /opt/program
RUN pip install -r requirements.txt

#RUN pip install opencv-python-headless

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY inference /opt/program

WORKDIR /opt/program