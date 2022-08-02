#!/bin/bash

docker run --gpus all \
-v /home/user:/workspace/home \
-e OMPI_ALLOW_RUN_AS_ROOT="1" \
-e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1" \
-e DISTDL_DEVICE="GPU" \
-it ai4fluidflow-train:v1.0