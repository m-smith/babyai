#!/bin/bash

gpu=$1

DIR=/project

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

NV_GPU="$gpu" ${cmd} run -it --rm\
        --net host \
        --name it-babyai-up-$gpu \
        -v `pwd`/:$DIR:rw \
        -t babyai-up-$USER \
        ${@:2}
