#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

#!/bin/bash

docker run --gpus all \
-v /home/user:/workspace/home \
-e OMPI_ALLOW_RUN_AS_ROOT="1" \
-e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1" \
-e DISTDL_DEVICE="GPU" \
-it ai4fluidflow-train:v1.0