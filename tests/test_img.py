import torch
import torchvision.transforms.functional as tf
from PIL import Image
import numpy as np

gt_path = r"/data/new_disk5/sam/data/RemoteSensing/SKD_Xingzheng/gaussian_data/block_insta_five/insta_gaussian_image_five_group0/images/1693119526200000000_D.png"

gt_img = Image.open(gt_path)
print("Done")
gt = tf.to_tensor(gt_img).unsqueeze(0)[:, :3, :, :].cuda()
print(np.array(gt_img).shape)
if np.array(gt_img).shape[2] == 4:

    mask = tf.to_tensor(gt_img)[3, :, :].repeat(3, 1, 1).unsqueeze(0).cuda()

print(mask.shape)
mask_img = mask.squeeze(0).cpu()
pil_image = tf.to_pil_image(mask_img)
pil_image.save("mask.png")