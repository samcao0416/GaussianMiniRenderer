#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim, ssim_with_mask
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr, psnr_with_mask
from argparse import ArgumentParser
from glob import glob
import numpy as np

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / (fname[:-4] + ".jpg"))
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        # renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :])
        # gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :])
        image_names.append(fname)
    return renders, gts, image_names

def read_colmap_extrinsics(path):
    image_names = []
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9][:-4]
                image_names.append(image_name)
                try:
                    elems = fid.readline().split()
                    xys = np.column_stack([tuple(map(float, elems[0::3])),
                                        tuple(map(float, elems[1::3]))])
                    point3D_ids = np.array(tuple(map(int, elems[2::3])))
                except:
                    xys = None
                    point3D_ids = None
                # images[image_id] = Image(
                #     id=image_id, qvec=qvec, tvec=tvec,
                #     camera_id=camera_id, name=image_name,
                #     xys=xys, point3D_ids=point3D_ids)
    return image_names

def evaluate(render_path, gt_path, mask_path):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}

    
    # try:
    print("Scene:", render_path)
    full_dict[render_path] = {}
    per_view_dict[render_path] = {}
    full_dict_polytopeonly[render_path] = {}
    per_view_dict_polytopeonly[render_path] = {}


        # full_dict[render_path][method] = {}
        # per_view_dict[render_path][method] = {}
        # full_dict_polytopeonly[render_path][method] = {}
        # per_view_dict_polytopeonly[render_path][method] = {}

    gt_dir = gt_path + "/images"
    gt_image_txt = gt_path + "/sparse/0/images.txt"
    renders_dir = render_path  + "/renders"

    image_names = read_colmap_extrinsics(gt_image_txt)
    # mask_dir = Path(gt_path) / "mask.png"
    # print(renders_dir, gt_dir)
    # raise Exception
    # renders, gts, image_names = readImages(renders_dir, gt_dir)
    ssims = []
    psnrs = []
    lpipss = []

    gt_img_path_list = sorted(glob(gt_dir + "/*.jpg"))
    if gt_img_path_list == []:
        gt_img_path_list = sorted(glob(gt_dir + "/*.png"))

    new_list = []
    for gt_img_path in gt_img_path_list:

        # print(os.path.basename(gt_img_path))
        img_name = os.path.basename(gt_img_path)[:-4]
        if img_name in image_names:
            # gt_img_path_list.remove(gt_img_path)
            new_list.append(gt_img_path)

    # print(cnt)
    gt_img_path_list = new_list

    render_img_path_list = sorted(glob(renders_dir + "/*.png"))
    if render_img_path_list == []:
        render_img_path_list = sorted(glob(renders_dir + "/*.jpg"))

    if len(gt_img_path_list) != len(render_img_path_list):
        print(len(gt_img_path_list))
        print(len(render_img_path_list))
        raise ValueError("Number of images in the ground truth and render directories do not match")
    
    if mask_path is not None:
        mask_img_path_list = sorted(glob(mask_path + "/*.png"))
        new_list = []
        for mask_img_path in mask_img_path_list:

            # print(os.path.basename(gt_img_path))
            img_name = os.path.basename(mask_img_path)[:-4]
            if img_name in image_names:
                # gt_img_path_list.remove(gt_img_path)
                new_list.append(mask_img_path)
        mask_img_path_list = new_list


    else:
        mask_img_path_list = None

    if (mask_img_path_list is not None) and (len(gt_img_path_list) != len(mask_img_path_list)):
        raise ValueError("Number of images in the ground truth and mask directories do not match")    

    for idx in tqdm(range(len(gt_img_path_list))):
        
        # check img_names
        gt_img_path = gt_img_path_list[idx]
        render_img_path = render_img_path_list[idx]

        if os.path.basename(gt_img_path)[:-4] != os.path.basename(render_img_path)[:-4]:
            raise ValueError("GT and Render Image names do not match")
        
        gt_img = Image.open(gt_img_path)
        gt = tf.to_tensor(gt_img).unsqueeze(0)[:, :3, :, :].cuda()    

        render_img = Image.open(render_img_path)
        render = tf.to_tensor(render_img).unsqueeze(0)[:, :3, :, :].cuda()

        if np.array(gt_img).shape[2] == 4:

            mask = tf.to_tensor(gt_img)[3, :, :].repeat(3, 1, 1).unsqueeze(0).cuda()

        elif mask_img_path_list is not None:
            mask_img_path = mask_img_path_list[idx]

            if os.path.basename(gt_img_path)[:-4] != os.path.basename(mask_img_path)[:-4]:
                raise ValueError("GT and Mask Image names do not match")
            
            mask = tf.to_tensor(Image.open(mask_img_path)).unsqueeze(0)[:, :3, :, :].cuda()
            if mask.shape[1] == 1:
                mask = mask.repeat(1, 3, 1, 1)
        else:
            mask = torch.ones_like(gt)        

        # renders[idx] = renders[idx] * mask
        # gts[idx] = gts[idx] * mask
            
        if psnr_with_mask(render, gt, mask) > 0:
            ssims.append(ssim_with_mask(render, gt, mask))
            psnrs.append(psnr_with_mask(render, gt, mask))
            lpipss.append(lpips(render * mask, gt * mask, net_type='vgg'))
        # if psnr(renders[idx], gts[idx]) > 15:
        #     ssims.append(ssim(renders[idx], gts[idx]))
        #     psnrs.append(psnr(renders[idx], gts[idx]))
        #     lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
    print("")

    full_dict[render_path].update({"SSIM": torch.tensor(ssims).mean().item(),
                                            "PSNR": torch.tensor(psnrs).mean().item(),
                                            "LPIPS": torch.tensor(lpipss).mean().item()})
    per_view_dict[render_path].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

    with open(render_path + "/results.json", 'w') as fp:
        json.dump(full_dict[render_path], fp, indent=True)
    with open(render_path + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[render_path], fp, indent=True)
    # except:
    #     print("Unable to compute metrics for model", render_path)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    # parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--render', '-r', required=True, type=str, default=None)
    parser.add_argument('--gt', '-g', required=True, type=str, default=None)
    parser.add_argument('--mask', '-m', type=str, default=None)
    args = parser.parse_args()
    # evaluate(args.model_paths, args.gt)
    evaluate(args.render, args.gt, args.mask)
