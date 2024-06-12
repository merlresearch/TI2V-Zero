# Copyright (c) 2024 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: Apache-2.0
#
# Code adapted from https://github.com/modelscope/modelscope/blob/57791a8cc59ccf9eda8b94a9a9512d9e3029c00b/modelscope/models/cv/anydoor/ldm/util.py -- Apache-2.0 license

import importlib
import os
import random
from copy import deepcopy
from inspect import isfunction

import cv2
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype("data/DejaVuSans.ttf", size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start : start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def center_crop(img, new_width=None, new_height=None):
    width = img.shape[1]
    height = img.shape[0]

    if width == height:
        return img

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def resize(img_npy, IMG_H, IMG_W):
    return np.asarray(Image.fromarray(img_npy).resize((IMG_H, IMG_W)))


def preprocess_image(img):
    img_tensor = torch.from_numpy(img / 255.0).type(torch.float32)
    img_tensor = img_tensor.unsqueeze(dim=0)
    img_tensor = img_tensor.permute(0, 3, 1, 2)  # nchw
    # normalization
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    mean = torch.tensor(mean, device=img_tensor.device).reshape(1, -1, 1, 1)  # nchw
    std = torch.tensor(std, device=img_tensor.device).reshape(1, -1, 1, 1)  # nchw
    img_tensor = img_tensor.sub_(mean).div_(std)
    return img_tensor


def postprocess_image(img_tensor, batch_idx=0):
    img_tensor = img_tensor.clone().detach().cpu()
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    mean = torch.tensor(mean, device=img_tensor.device).reshape(1, -1, 1, 1)  # nchw
    std = torch.tensor(std, device=img_tensor.device).reshape(1, -1, 1, 1)  # nchw
    img_tensor = img_tensor.mul_(std).add_(mean)
    img_tensor = img_tensor[batch_idx].permute(1, 2, 0)
    img_tensor[img_tensor < 0] = 0
    img_tensor[img_tensor > 1] = 1
    img_data = np.array(img_tensor * 255, dtype=np.uint8)
    return img_data


def resize_with_border(im, desired_size, interpolation):
    old_size = im.shape[:2]
    ratio = float(desired_size) / max(old_size)
    new_size = tuple(int(x * ratio) for x in old_size)

    im = cv2.resize(im, (new_size[1], new_size[0]), interpolation=interpolation)
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_im


def nearest_true_index(mask, index):
    if mask[index]:
        return index  # The given index is True, so it's the nearest True index.

    left_index = index - 1
    right_index = index + 1

    while left_index >= 0 or right_index < len(mask):
        if left_index >= 0 and mask[left_index]:
            return left_index
        if right_index < len(mask) and mask[right_index]:
            return right_index

        left_index -= 1
        right_index += 1

    return None  # No True value found in the mask.


def binary_to_hex(binary_list):
    binary_list = deepcopy(binary_list)
    # binary_list.reverse()  # Reverse the list to match bit order.
    binary_string = "".join([str(int(bit)) for bit in binary_list])
    decimal_number = int(binary_string, 2)
    hex_string = hex(decimal_number).lstrip("0x")
    return hex_string


def list2gif(img_path_list, gif_path, save_img_dir):
    save_npy_list = [imageio.v2.imread(x) for x in img_path_list]
    imageio.mimwrite(gif_path, save_npy_list, duration=1000 / 8)
    for i, save_npy in enumerate(save_npy_list):
        imageio.v2.imsave(os.path.join(save_img_dir, "%04d.png" % i), save_npy)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
