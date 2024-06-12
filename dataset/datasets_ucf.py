# Copyright (c) 2024 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2023 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: BSD-2-Clause
#
# Code adapted from https://github.com/nihaomiao/CVPR23_LFDM/tree/main/preprocessing -- BSD-2-Clause License

# For video generation on UCF dataset
import os
import sys

sys.path.append(os.getcwd())
import glob

import cv2
import imageio
import numpy as np
import torch
import torch.utils.data as data

from util import center_crop, preprocess_image, resize, setup_seed


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            frame_list.append(frame[:, :, ::-1])
        else:
            break
    cap.release()
    return frame_list


def video_to_img_dir(video_path, img_dir_path):
    frame_list = read_video(video_path)
    video_name = os.path.basename(video_path)[:-4]
    sub_img_dir_path = os.path.join(img_dir_path, video_name)
    os.makedirs(sub_img_dir_path, exist_ok=True)
    num_frame = len(frame_list)
    for i in range(num_frame):
        frame_name = "%s-%03d.png" % (video_name, i)
        frame_path = os.path.join(sub_img_dir_path, frame_name)
        imageio.imsave(frame_path, frame_list[i])


def preprocess_ucf101_with_selected_class():
    selected_class_dict = {
        "ApplyEyeMakeup": "A person is applying eye makeup.",
        "BabyCrawling": "A Baby is crawling.",
        "BreastStroke": "A person is performing breaststroke.",
        "Drumming": "A person is drumming.",
        "HorseRiding": "A person is riding horse.",
        "Kayaking": "A person is kayaking.",
        "PlayingGuitar": "A person is playing Guitar.",
        "Surfing": "A person is surfing.",
        "ShavingBeard": "A person is shaving beard.",
    }
    root_dir = "/data/hfn5052/img2vid-zero/dataset/UCF-101"
    save_dir = "/data/hfn5052/img2vid-zero/dataset/UCF-101-frames"

    for class_name in list(selected_class_dict.keys()):
        cur_save_dir = os.path.join(save_dir, class_name)
        os.makedirs(cur_save_dir, exist_ok=True)
        video_list_path = os.path.join(root_dir, class_name)
        video_list = os.listdir(video_list_path)
        video_path_list = [os.path.join(root_dir, class_name, x) for x in video_list]
        for video_path in video_path_list:
            video_to_img_dir(video_path, cur_save_dir)
            print(video_path)


# for consistently generating videos
class UCF_gen(data.Dataset):
    def __init__(self, data_dir, num_frames=16, image_size=256):
        super(UCF_gen, self).__init__()
        self.selected_class_dict = {
            "ApplyEyeMakeup": "A person is applying eye makeup.",
            "BabyCrawling": "A Baby is crawling.",
            "BreastStroke": "A person is performing breaststroke.",
            "Drumming": "A person is drumming.",
            "HorseRiding": "A person is riding horse.",
            "Kayaking": "A person is kayaking.",
            "PlayingGuitar": "A person is playing Guitar.",
            "Surfing": "A person is surfing.",
            "ShavingBeard": "A person is shaving beard.",
        }
        self.num_scene = 10
        self.num_combs = len(self.selected_class_dict) * self.num_scene
        self.num_frames = num_frames
        assert self.num_frames == 16
        self.image_size = image_size
        self.video_dict = {}
        for class_name in list(self.selected_class_dict.keys()):
            self.video_dict[class_name] = {}
            class_video_dir_path = os.path.join(data_dir, class_name)
            for scene_idx in range(self.num_scene):
                video_name = "v_%s_g%02d_c*" % (class_name, scene_idx + 1)
                cur_video_path_name = os.path.join(class_video_dir_path, video_name)
                cur_video_path_list = glob.glob(cur_video_path_name)
                cur_video_path_list.sort()
                assert len(cur_video_path_list)
                self.video_dict[class_name][scene_idx + 1] = cur_video_path_list

    def __len__(self):
        return int(self.num_combs)

    def __getitem__(self, index):
        class_idx = index % len(self.selected_class_dict)
        scene_idx = index // self.num_scene
        class_name = list(self.selected_class_dict.keys())[class_idx]
        scene_name = scene_idx + 1
        video_path_list = self.video_dict[class_name][scene_name]
        if len(video_path_list) == 0:
            raise NotImplementedError
        else:
            video_path = str(np.random.choice(video_path_list, size=1)[0])
            video_name = os.path.basename(video_path)
        frame_name_list = os.listdir(video_path)
        frame_name_list = [x for x in frame_name_list if x.endswith("jpg") or x.endswith("png")]
        frame_name_list.sort()
        frame_path_list = [os.path.join(video_path, frame_name) for frame_name in frame_name_list]
        # sampling
        if len(frame_path_list) > 48:
            frame_path_list = frame_path_list[::3]
            frame_path_list = frame_path_list[:16]
        else:
            frame_path_list = frame_path_list[:16]
        # read image
        sample_frame_list = [imageio.v2.imread(x) for x in frame_path_list]
        sample_frame_list = [center_crop(x) for x in sample_frame_list]
        sample_frame_list = [resize(x, self.image_size, self.image_size) for x in sample_frame_list]
        sample_frame_list_tensor = [preprocess_image(x) for x in sample_frame_list]
        sample_frame_list_tensor = torch.cat(sample_frame_list_tensor, dim=0)
        sample_frame_list_tensor = sample_frame_list_tensor.permute(1, 0, 2, 3)

        # design text prompt
        prompt = self.selected_class_dict[class_name]

        return sample_frame_list_tensor, prompt, video_name


if __name__ == "__main__":
    # use preprocess_ucf101_with_selected_class() to sample frames from video on UCF101
    # preprocess_ucf101_with_selected_class()
    data_dir = "/data/hfn5052/img2vid-zero/dataset/UCF-101-frames"
    setup_seed(1234)
    dataset = UCF_gen(data_dir)
    for i in range(len(dataset)):
        output = dataset[i]
        print(output[1], output[2])
