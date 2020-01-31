# Some of our servers will not have more recent CUDA version on them
FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04
#FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

### --build-args
ARG UID=1000

### CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Generic Installs
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         sudo \
         vim \
 	 tmux \
  	 qt5-default\
         qttools5-dev-tools\
         ca-certificates \
         libjpeg-dev \
         python-opencv \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*

# Anaconda Installs
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install python=3.6 && \
     /opt/conda/bin/conda install numpy pyyaml scipy ipython cython mkl mkl-include && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH


# Install dependencies
RUN pip install --upgrade pip wheel
RUN pip install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
RUN pip install torchvision
RUN pip install gym sacred GitPython pymongo opencv-python plumbum blosc pyqt5
RUN pip install -e git+https://github.com/maximecb/gym-minigrid#egg=gym-minigrid


# Section to get permissions right, and avoid running inside as root {{
# Create a user matching the UID, and create/chmod home dir (== project directory)
# <<UID>> will be replaced by user id
RUN useradd -d /project -u $UID --create-home user && echo "user:pass" | chpasswd && adduser user sudo

USER user
WORKDIR /project/
ADD . /project/
ENV PYTHONPATH "$PYTHONPATH:/project/"

# RUN pip install --editable .
# Copy models into the docker image
# COPY models models/
