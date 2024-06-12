# Copyright (c) 2024 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2023 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: BSD-2-Clause
#
# Code adapted from https://github.com/nihaomiao/CVPR23_LFDM/tree/main/demo -- BSD-2-Clause License

# keep generating videos on MUG dataset for evaluation
import os
import sys

import torch

sys.path.append(os.getcwd())
import argparse
import timeit

import imageio
import numpy as np
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.utils import data
from tqdm import tqdm

from dataset.datasets_mug import MUG_gen
from modelscope_t2v_pipeline import TextToVideoSynthesisPipeline, tensor2vid
from util import AverageMeter, postprocess_image, setup_seed

start = timeit.default_timer()

# PARAMETER SETTINGS
BATCH_SIZE = 1
GPU = "1"
# set path to save outputs
root_dir = "/data/hfn5052/img2vid-zero/cvpr24/show/MUG"
# set path to MUG dataset
data_dir = "/data/hfn5052/text2motion/dataset/MUG"
# set #DDIM and #Resample
ddim_step = 10
resample_iter = 4
# set the number of new frames
NUM_NEW_FRAMES = 15
# set the number of generated videos
NUM_VIDEOS = 1000
# After running initialization.py, set the config path to your ModelScope path
config = {"model": "./weights", "device": "gpu"}

postfix = "-resample%02d-s%02d" % (resample_iter, ddim_step)
add_vid_cond = True
use_ddpm_inversion = True
IMG_SIZE = 256
NUM_COND_FRAMES = 15
NUM_FRAMES = 16
RANDOM_SEED = 1234
NUM_ITER = NUM_VIDEOS // BATCH_SIZE
print(postfix)
print("batch size:", BATCH_SIZE)
print("video_cond:", add_vid_cond, "ddpm_inv:", use_ddpm_inversion)
print("#new_frame:", NUM_NEW_FRAMES)
print(config)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="TI2V-Zero")
    parser.add_argument("--num-workers", default=0)
    parser.add_argument("--gpu", default=GPU, help="choose gpu device.")
    parser.add_argument("--print-freq", "-p", default=1, type=int, metavar="N", help="print frequency")
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Number of images sent to the network in one step."
    )
    parser.add_argument(
        "--random-seed", type=int, default=RANDOM_SEED, help="Random seed to have reproducible results."
    )
    return parser.parse_args()


args = get_arguments()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.enabled = True
    cudnn.benchmark = True
    setup_seed(args.random_seed)

    t2v_pipeline = TextToVideoSynthesisPipeline(**config)

    setup_seed(args.random_seed)
    testloader = data.DataLoader(
        MUG_gen(data_dir=data_dir, image_size=IMG_SIZE, num_frames=NUM_FRAMES),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    batch_time = AverageMeter()
    data_time = AverageMeter()

    iter_end = timeit.default_timer()
    cnt = 0

    global_iter = 0
    video_dir = os.path.join(root_dir, "gen" + postfix)
    os.makedirs(video_dir, exist_ok=True)
    image_dir = os.path.join(root_dir, "gen_img" + postfix)
    os.makedirs(image_dir, exist_ok=True)

    while global_iter < NUM_ITER:
        for i_iter, batch in enumerate(testloader):
            data_time.update(timeit.default_timer() - iter_end)
            real_vids, ref_texts, real_names = batch
            # use first frame of each video as reference frame
            ref_imgs = real_vids[:, :, 0, :, :].clone().detach()
            bs = real_vids.size(0)
            batch_time.update(timeit.default_timer() - iter_end)
            vid_tensor = ref_imgs.unsqueeze(dim=2).repeat(1, 1, NUM_COND_FRAMES, 1, 1)
            new_output_tensor = vid_tensor.clone().detach().cpu()

            processed_input = t2v_pipeline.preprocess(ref_texts)

            for frame_iter in tqdm(range(NUM_NEW_FRAMES)):
                output = t2v_pipeline.forward_with_vid_resample(
                    processed_input,
                    vid=vid_tensor,
                    add_vid_cond=add_vid_cond,
                    use_ddpm_inversion=use_ddpm_inversion,
                    resample_iter=resample_iter,
                    ddim_step=ddim_step,
                    guide_scale=9.0,
                )
                with torch.no_grad():
                    new_frame = t2v_pipeline.model.autoencoder.decode(output[:, :, -1].cuda())
                new_frame = new_frame.data.cpu().unsqueeze(dim=2)
                new_output_tensor = torch.cat((new_output_tensor, new_frame), dim=2)
                vid_tensor = new_output_tensor[:, :, (frame_iter + 1) :]
                assert vid_tensor.size(2) == NUM_COND_FRAMES

            for batch_idx in range(bs):
                output_video = tensor2vid(new_output_tensor[batch_idx, :, (NUM_COND_FRAMES - 1) :].unsqueeze(dim=0))
                msk_size = ref_imgs.shape[-1]
                save_src_img = postprocess_image(ref_imgs, batch_idx)
                nf = real_vids.size(2)
                assert nf == NUM_FRAMES
                new_im_list = []

                img_dir_name = "%04d_%s" % (cnt, real_names[batch_idx])
                cur_img_dir = os.path.join(image_dir, img_dir_name)
                os.makedirs(cur_img_dir, exist_ok=True)

                for frame_idx in range(nf):
                    save_real_img = postprocess_image(real_vids[:, :, frame_idx], batch_idx)
                    save_out_img = output_video[frame_idx]
                    new_im = Image.new("RGB", (msk_size * 3, msk_size))
                    new_im.paste(Image.fromarray(save_src_img, "RGB"), (0, 0))
                    new_im.paste(Image.fromarray(save_real_img, "RGB"), (msk_size, 0))
                    new_im.paste(Image.fromarray(save_out_img, "RGB"), (msk_size * 2, 0))
                    new_im_arr = np.array(new_im)
                    # save frame
                    new_im_name = "%03d_%04d_%s.png" % (frame_idx, cnt, real_names[batch_idx])
                    imageio.imsave(os.path.join(cur_img_dir, new_im_name), new_im_arr)
                    new_im_list.append(new_im_arr)
                video_name = "%04d_%s.gif" % (cnt, real_names[batch_idx])
                print(video_name)
                imageio.mimwrite(os.path.join(video_dir, video_name), new_im_list, duration=1000 / 8)
                cnt += 1

            iter_end = timeit.default_timer()

            if global_iter % args.print_freq == 0:
                print(
                    "Test:[{0}/{1}]\t"
                    "Time {batch_time.val:.3f}({batch_time.avg:.3f})".format(
                        global_iter, NUM_ITER, batch_time=batch_time
                    )
                )
            global_iter += 1

    end = timeit.default_timer()
    print(end - start, "seconds")
    print(video_dir)
    print(image_dir)


if __name__ == "__main__":
    main()
