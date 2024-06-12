# Copyright (c) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Download pretrained modelscope models
# Code example from ModelScope: https://huggingface.co/ali-vilab/modelscope-damo-text-to-video-synthesis

import pathlib

from huggingface_hub import snapshot_download

model_dir = pathlib.Path("./weights")
snapshot_download("damo-vilab/modelscope-damo-text-to-video-synthesis", repo_type="model", local_dir=model_dir)
