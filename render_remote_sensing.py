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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.camera_utils import cameraList_from_camInfos
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.dataset_readers import readNovelCameras
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text

def render_set(out, iteration, views, gaussians, pipeline, background, split_list):
    print("images saved in: ",out)
    # train_render_path = os.path.join(out, "train", "ours_{}".format(iteration), "renders")
    # test_render_path = os.path.join(out, "test", "ours_{}".format(iteration), "renders")
    # novel_D_path = os.path.join(out, "novel_D", "ours_{}".format(iteration), "renders")
    demo_video_path = os.path.join(out, "renders")


    # makedirs(train_render_path, exist_ok=True)
    # makedirs(test_render_path, exist_ok=True)
    # makedirs(novel_D_path, exist_ok=True)
    makedirs(demo_video_path, exist_ok=True)

    for idx, view in enumerate(tqdm(split_list, desc="Rendering demo progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(demo_video_path, view.image_name + ".png"))

    # for idx, view in enumerate(tqdm(split_list, desc="Rendering test progress")):
    #     rendering = render(view, gaussians, pipeline, background)["render"]
    #     torchvision.utils.save_image(rendering, os.path.join(test_render_path, view.image_name + ".png"))
        
    # for idx, view in enumerate(tqdm(views, desc="Rendering train progress")):
    #     rendering = render(view, gaussians, pipeline, background)["render"]
    #     torchvision.utils.save_image(rendering, os.path.join(train_render_path, view.image_name + ".png"))



def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, split : str, out : str, cameras : str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if (cameras is None) or (not ( os.path.exists(cameras))):

            cameras_intrinsic_file = os.path.join(dataset.source_path, "sparse/0", "cameras.txt")
        
        else:
            cameras_intrinsic_file = cameras

        cameras_extrinsic_file = split
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        split_info_unsorted = readNovelCameras(cam_extrinsics, cam_intrinsics)
        split_infos = sorted(split_info_unsorted.copy(), key = lambda x : x.image_name)
        args.resolution = 1
        # args.data_device = "cpu"
        split_list = cameraList_from_camInfos(split_infos, 1.0, args)

        render_set(out, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, split_list)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--split", default=None, type=str, help="path to novel render txt file path")
    parser.add_argument("--out", default=None, type=str)
    parser.add_argument("--cameras", default=None, type=str, help= "path to used camera.txt")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.split, args.out, args.cameras)