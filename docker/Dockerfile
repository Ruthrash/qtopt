ARG image=nvidia/cudagl:11.3.0-devel-ubuntu20.04
FROM ${image}

ARG DEBIAN_FRONTEND=noninteractive
ENV WS_DIR="/root/git"
WORKDIR ${WS_DIR}

SHELL ["/bin/bash", "-c"]
# Dependencies for latency plot
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    curl \
    grep \
    build-essential \
    make \
    libnuma-dev \
    python3 \
    python3-distutils \
    gnuplot-qt \ 
    sshpass \
    iproute2 \
    python3-pip \
    gnome-terminal\
    nano\
    vim
# Dependencies for Bash unit-tests
RUN apt-get update && apt-get install -y \
    bats \
    dialog \
    tmux \
    iputils-ping 	

RUN pip3 install gym \ 
    torch torch-vision \ 
    IPython \
    matplotlib
# RUN if [[ "x$build_contactgraspnet" == "x1" ]] ;\
#     then echo "build_contactgraspnet is set to TRUE" \
#     && echo "Building conda for contact_graspnet" \
#     && apt-get update && apt-get upgrade -y && apt-get install -y  freeglut3-dev\
#     && export CONDA_DIR=/opt/conda \
#     && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh  \
#     && /bin/bash ~/miniconda.sh -b -p /opt/conda \ 
#     && export PATH=$CONDA_DIR/bin:$PATH \ 
#     && cd /root/git \ 
#     && git clone https://github.com/NVlabs/contact_graspnet \
#     && cd contact_graspnet\
#     && pip3 install open3d pyrealsense2\
#     && conda env create -f contact_graspnet_env.yml \
#     && conda init bash ; \
#     fi

    
# ENV PATH /opt/conda/bin:$PATH




SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]


