# Copyright (c) 2024 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2023 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: BSD-2-Clause
#
# Code adapted from https://github.com/nihaomiao/CVPR23_LFDM/tree/main/preprocessing -- BSD-2-Clause License

# For video generation on MUG dataset
import os
import sys

sys.path.append(os.getcwd())
import imageio
import numpy as np
import torch
import torch.utils.data as data

from util import center_crop, preprocess_image, resize


# for consistently generating videos
class MUG_gen(data.Dataset):
    def __init__(self, data_dir, num_frames=16, image_size=256):
        super(MUG_gen, self).__init__()
        self.female_ID = ["001", "002", "006", "046", "048"]
        self.male_ID = ["007", "010", "013", "014", "020"]
        self.female_prompt = "A woman with the expression of slight * on her face."
        self.male_prompt = "A man with the expression of slight * on his face."
        self.test_ID = self.female_ID + self.male_ID
        session_ID = ["002", "003", "049"]
        self.exp_list = ["anger", "happiness", "sadness", "surprise"]
        self.num_combs = len(self.test_ID) * len(self.exp_list)
        self.num_frames = num_frames
        assert self.num_frames == 16
        self.image_size = image_size
        self.video_path_list = []
        for video_name in self.test_ID:
            if video_name not in session_ID:
                for exp_name in self.exp_list:
                    cur_video_dir_path = os.path.join(data_dir, video_name, exp_name)
                    if os.path.exists(cur_video_dir_path):
                        cur_video_name_list = os.listdir(cur_video_dir_path)
                        cur_video_name_list.sort()
                        for cur_video_name in cur_video_name_list:
                            cur_video_path = os.path.join(cur_video_dir_path, cur_video_name)
                            self.video_path_list.append(cur_video_path)
            else:
                for session_name in ["session0", "session0"]:
                    for exp_name in self.exp_list:
                        cur_video_dir_path = os.path.join(data_dir, video_name, session_name, exp_name)
                        if os.path.exists(cur_video_dir_path):
                            cur_video_name_list = os.listdir(cur_video_dir_path)
                            cur_video_name_list.sort()
                            for cur_video_name in cur_video_name_list:
                                cur_video_path = os.path.join(cur_video_dir_path, cur_video_name)
                                self.video_path_list.append(cur_video_path)
        # group each video according to subject and expression
        self.video_dict = {}
        for comb_idx in range(self.num_combs):
            sub_idx = comb_idx % len(self.test_ID)
            exp_idx = comb_idx // len(self.test_ID)
            sub_name = self.test_ID[sub_idx]
            exp_name = self.exp_list[exp_idx]
            if sub_name not in self.video_dict.keys():
                self.video_dict[sub_name] = {}
            self.video_dict[sub_name][exp_name] = []

        for video_path in self.video_path_list:
            sub_name = video_path.split("/")[6]
            exp_name = video_path.split("/")[-2]
            assert sub_name in self.test_ID
            assert exp_name in self.exp_list
            self.video_dict[sub_name][exp_name].append(video_path)

    def __len__(self):
        return int(self.num_combs)

    def __getitem__(self, index):
        sub_idx = index % len(self.test_ID)
        exp_idx = index // len(self.test_ID)
        sub_name = self.test_ID[sub_idx]
        exp_name = self.exp_list[exp_idx]
        video_path_list = self.video_dict[sub_name][exp_name]
        if len(video_path_list) == 0:
            raise NotImplementedError
        else:
            video_path = str(np.random.choice(video_path_list, size=1)[0])
            video_name = (
                "_".join(video_path.split("/")[-3:])
                if "session" not in video_path
                else "_".join(video_path.split("/")[-4:])
            )
        exp_name = video_name.split("_")[-2]
        assert exp_name in self.exp_list
        frame_name_list = os.listdir(video_path)
        frame_name_list = [x for x in frame_name_list if x.endswith("jpg") or x.endswith("png")]
        frame_name_list.sort()
        frame_path_list = [os.path.join(video_path, frame_name) for frame_name in frame_name_list]
        # sampling
        if len(frame_path_list) > 96:
            frame_path_list = frame_path_list[::6]
            frame_path_list = frame_path_list[:16]
        else:
            sample_idx_list = np.linspace(start=0, stop=len(frame_path_list) - 1, num=self.num_frames, dtype=int)
            frame_path_list = [frame_path_list[x] for x in sample_idx_list]
        # read image
        sample_frame_list = [imageio.v2.imread(x) for x in frame_path_list]
        sample_frame_list = [center_crop(x) for x in sample_frame_list]
        sample_frame_list = [resize(x, self.image_size, self.image_size) for x in sample_frame_list]
        sample_frame_list_tensor = [preprocess_image(x) for x in sample_frame_list]
        sample_frame_list_tensor = torch.cat(sample_frame_list_tensor, dim=0)
        sample_frame_list_tensor = sample_frame_list_tensor.permute(1, 0, 2, 3)

        # design text prompt
        if sub_name in self.female_ID:
            prompt = self.female_prompt.replace("*", exp_name)
        else:
            prompt = self.male_prompt.replace("*", exp_name)

        return sample_frame_list_tensor, prompt, video_name


if __name__ == "__main__":
    data_dir = "/data/hfn5052/text2motion/dataset/MUG"
    dataset = MUG_gen(data_dir)

    test_dataset = data.DataLoader(dataset=dataset, batch_size=2, num_workers=0, shuffle=False)
    for i, batch in enumerate(test_dataset):
        print(i)
