import sys
sys.path.append("../")
import os

data_name_list = ["SKD_Xingzheng", "STAR_Center", "underground1010"]
secondary_name_list = ['block_insta_five', 'block_insta_single', 'block_titan']
thirdary_name_list = ['insta_gaussian_image_five_group', 'insta_gaussian_image_single_group', 'titan_gaussian_group']
group_num = [2, 3, 5]

for index in range(len(data_name_list)):
    for group_id in range(group_num[index]):
        for j in range(len(secondary_name_list)):

            model_path = os.path.join("/data/new_disk5/sam/exps/RemoteSensing", data_name_list[index], "gaussian_data", secondary_name_list[j], thirdary_name_list[j]+str(group_id))
            source_path = os.path.join("/data/new_disk5/sam/data/RemoteSensing", data_name_list[index], "gaussian_data", secondary_name_list[j], thirdary_name_list[j]+str(group_id))
            mask_path = os.path.join(source_path, "masks")
            if not(os.path.exists(mask_path)):
                mask_path = None

            if os.path.exists(model_path):
                print(model_path)
                print("----------------------------------------------------------------------------")
                for k in range(len(secondary_name_list)):
                    novel_view_folder = os.path.join("/data/new_disk5/sam/data/RemoteSensing", data_name_list[index], "gaussian_data", secondary_name_list[k], thirdary_name_list[k]+str(group_id))
                    # print("   ", novel_view_folder)
                    novel_extrinsics_path = os.path.join(novel_view_folder, "sparse/0/images.txt")
                    novel_intrinsics_path = os.path.join(novel_view_folder, "sparse/0/cameras.txt")
                    output_path = os.path.join("/data/new_disk5/sam/results/RemoteSensing", data_name_list[index], secondary_name_list[j], thirdary_name_list[j]+str(group_id), secondary_name_list[k])
                    # print("      ", output_path)

                    # os.system("python render_remote_sensing.py -s %s -m %s --split %s --cameras %s --out %s" %(source_path, model_path, novel_extrinsics_path, novel_intrinsics_path, output_path))
                    os.system("python metrics_with_mask.py -g %s -r %s -m %s" %(source_path, output_path, mask_path))

                print("----------------------------------------------------------------------------")
