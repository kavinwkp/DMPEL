import torch
import numpy as np
import cv2
import argparse
import h5py
import os
from functools import partial
import sys
# print(sys.path)
# sys.path.append('../dinov2')
# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'dinov2'))
# sys.path.append(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'dinov2'), 'dinov2'))
# print(sys.path)
sys.path.append('/public/home/group_luoping/leiyuheng/project/pearl/libero/lifelong/algos/lotus_skill_learning/dinov2')
from einops import rearrange
from easydict import EasyDict
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from dinov2.models import build_model_from_cfg
from dinov2.utils.config import setup

from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import get_autocast_dtype, build_model_for_eval
from dinov2.eval.utils import ModelWithIntermediateLayers

from sklearn.decomposition import PCA
from libero.lifelong.algos.lotus_skill_learning.models.model_utils import safe_cuda

# from utils.video_utils import KaedeVideoWriter

# Dataset_Name_List = [
#     "../datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo",
#     "../datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_demo",
#     "../datasets/libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo",
#     "../datasets/libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo",
#     "../datasets/libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo",
#     "../datasets/libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo",
#     "../datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo",
#     "../datasets/libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo",
#     "../datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo",
#     "../datasets/libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo",
# ]

# Dataset_Name_List = [
#     "../datasets/libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo",
#     "../datasets/libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo",
#     "../datasets/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo",
#     "../datasets/libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo",
#     "../datasets/libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo",
#     "../datasets/libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo",
#     "../datasets/libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo",
#     "../datasets/libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo",
#     "../datasets/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo",
#     "../datasets/libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo",
# ]

# Dataset_Name_List = [
#     "../datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/pick_up_the_cream_cheese_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/pick_up_the_ketchup_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/pick_up_the_butter_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/pick_up_the_milk_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket_demo",
# ]

# Dataset_Name_List = [
#     "../datasets/libero_goal/open_the_middle_drawer_of_the_cabinet_demo",
#     "../datasets/libero_goal/put_the_bowl_on_the_stove_demo",
#     "../datasets/libero_goal/put_the_wine_bottle_on_top_of_the_cabinet_demo",
#     "../datasets/libero_goal/open_the_top_drawer_and_put_the_bowl_inside_demo",
#     "../datasets/libero_goal/put_the_bowl_on_top_of_the_cabinet_demo",
#     "../datasets/libero_goal/push_the_plate_to_the_front_of_the_stove_demo",
#     "../datasets/libero_goal/put_the_cream_cheese_in_the_bowl_demo",
#     "../datasets/libero_goal/turn_on_the_stove_demo",
#     "../datasets/libero_goal/put_the_bowl_on_the_plate_demo",
#     "../datasets/libero_goal/put_the_wine_bottle_on_the_rack_demo",
# ]

# Dataset_Name_List = [
#     "../datasets/libero_90/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo",
#     "../datasets/libero_90/put_the_bowl_on_the_stove_demo",
#     "../datasets/libero_90/put_the_wine_bottle_on_top_of_the_cabinet_demo",
#     "../datasets/libero_90/open_the_top_drawer_and_put_the_bowl_inside_demo",
#     "../datasets/libero_90/put_the_bowl_on_top_of_the_cabinet_demo",
#     "../datasets/libero_90/push_the_plate_to_the_front_of_the_stove_demo",
#     "../datasets/libero_goal/put_the_cream_cheese_in_the_bowl_demo",
#     "../datasets/libero_goal/turn_on_the_stove_demo",
#     "../datasets/libero_goal/put_the_bowl_on_the_plate_demo",
#     "../datasets/libero_goal/put_the_wine_bottle_on_the_rack_demo",
# ]


class DinoV2ImageProcessor(object):
    def __init__(self, args=None):
        if args is None:
            self.args = EasyDict()
            self.args.output_dir = ''
            self.args.opts = []
            self.args.pretrained_weights = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
            self.args.config_file = "dinov2/dinov2/configs/eval/vitb14_pretrain.yaml"
            # self.args.dataset_name = "libero_10"
            # self.args.exp_name = "dinov2_libero_10_image_only"
            # self.args.moality_str = "dinov2_agentview_eye_in_hand"
            # self.args.feature_dim = 768 * 2
        else:
            self.args = args
        # print("*****")
        print(self.args)
        self.model, self.autocast_dtype = self.setup_and_build_model()
        self.n_last_blocks_list = [1, 4]
        self.n_last_blocks = max(self.n_last_blocks_list)
        self.autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=self.autocast_dtype)
        self.feature_model = ModelWithIntermediateLayers(self.model, self.n_last_blocks, self.autocast_ctx)

    @staticmethod
    def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
        for t, m, s in zip(x, mean, std):
            t.sub_(m)
            t.div_(s)
        return x

    def setup_and_build_model(self):
        cudnn.benchmark = True
        config = setup(self.args)
        model = build_model_for_eval(config, self.args.pretrained_weights)
        model.eval()
        autocast_dtype = get_autocast_dtype(config)
        return model, autocast_dtype

    def process_image(self, img):
        # img = cv2.imread(image_path)
        sizes = [448, 224]
        features = []
        max_size = max(sizes) // 14

        for size in sizes:
            img = cv2.resize(img, (size, size))
            img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.
            img_tensor = self.color_normalize(img_tensor)
            feature = self.feature_model(img_tensor)[-1][0]
            new_feat = torch.nn.functional.interpolate(rearrange(feature, 'b (h w) c -> b c h w', h=int(np.sqrt(feature.shape[1]))), (max_size, max_size), mode="bilinear", align_corners=True, antialias=True)
            new_feat = rearrange(new_feat, 'b c h w -> b h w c')
            features.append(new_feat.squeeze(0))

        features = torch.mean(torch.stack(features), dim=0)
        return features
        # return self.pca_transform(features, max_size)

    def process_images(self, imgs):
        # imgs should be a batch of images, shape (batch_size, height, width, channels)
        sizes = [448, 224]
        max_size = max(sizes) // 14
        batch_size = len(imgs)

        all_features = []
        for size in sizes:
            imgs_resized = [cv2.resize(img, (size, size)) for img in imgs]
            img_tensors = torch.stack([torch.tensor(img).permute(2, 0, 1).float() for img in imgs_resized]).cuda() / 255.
            img_tensors = torch.cat([self.color_normalize(img_tensor.unsqueeze(0)) for img_tensor in img_tensors])
            features = self.feature_model(img_tensors)[-1][0]
            new_feats = torch.nn.functional.interpolate(rearrange(features, 'b (h w) c -> b c h w', h=int(np.sqrt(features.shape[1]))), (max_size, max_size), mode="bilinear", align_corners=True, antialias=True)
            new_feats = rearrange(new_feats, 'b c h w -> b h w c')
            all_features.append(new_feats)

        all_features = torch.mean(torch.stack(all_features), dim=0)
        return all_features


    @staticmethod
    def pca_transform(features, max_size):
        pca = PCA(n_components=3)
        pca_tensor = pca.fit_transform(features.detach().cpu().numpy().reshape(-1, 768))
        pca_tensor = (pca_tensor - pca_tensor.min()) / (pca_tensor.max() - pca_tensor.min())    
        pca_tensor = (pca_tensor * 255).astype(np.uint8).reshape(max_size, max_size, 3)
        # pca_tensor = 2 * pca_tensor - 1
        pca_tensor = pca_tensor.reshape(max_size, max_size, 32)
        return pca_tensor

    def save_image(self, pca_tensor, out_path="dinov2_pca.png"):
        cv2.imwrite(out_path, pca_tensor)

def compute_affinity(feat_1_tuple, feat_2_tuple, temperature=1):
    feat_1, h, w = feat_1_tuple
    feat_2, h2, w2 = feat_2_tuple
    feat_1 = rearrange(feat_1, 'h w c -> (h w) c')
    feat_2 = rearrange(feat_2, 'h w c -> (h w) c')
    sim_matrix = torch.einsum("lc,sc->ls", feat_1, feat_2) / temperature
    aff = sim_matrix
    # aff = F.softmax(aff, dim=0)
    aff = aff.cpu().view(h, w, h2, w2)
    # compute softmax over the first two axes
    return aff

def rescale_feature_map(img_tensor, target_h, target_w, convert_to_numpy=True):
    img_tensor = torch.nn.functional.interpolate(img_tensor, (target_h, target_w))
    if convert_to_numpy:
        return img_tensor.cpu().numpy()
    else:
        return img_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_name',
        type=str,
        default="dinov2_libero_10_image_only",
    )
    parser.add_argument(
        '--feature_dim',
        type=int,
        default=768*2,
    )
    parser.add_argument(
        '--modality_str',
        type=str,
        default="dinov2_agentview_eye_in_hand",
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=1024
    )
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        default="libero_10"
    )  
    args = parser.parse_args()
    modality_str = args.modality_str
    feature_dim = args.feature_dim
    dinov2 = DinoV2ImageProcessor()

    for dataset in ["libero_90", "libero_goal", "libero_object", "libero_spatial", "libero_10"]:
        directory = "/public/home/group_luoping/leiyuheng/dataset/datasets/" + dataset
        dataset_name_list = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                print(os.path.join(root, file))
                dataset_name_list.append(os.path.join(root, file))
        
        for dataset_name in dataset_name_list:
            dataset_name = dataset_name[:-5]
            dataset_hdf5_file = dataset_name + ".hdf5"
            # print(dataset_hdf5_file)
            f = h5py.File(dataset_hdf5_file, "r")
            demo_num = len(f['data'].keys())

            dataset_name_parts = dataset_name.split("/")
            part_2 = dataset_name_parts[-2]
            part_1 = dataset_name_parts[-1]
            embedding_name = f"/public/home/group_luoping/leiyuheng/dataset/datasets/lotus/dinov2_{dataset}_image_only/repr/{part_2}/{part_1}/embedding_{modality_str}_{feature_dim}.hdf5"
            os.makedirs(os.path.dirname(embedding_name), exist_ok=True)
            print("Saving embedding to", embedding_name)
            embedding_file = h5py.File(embedding_name, "w")
            grp = embedding_file.create_group("data")

            for i in tqdm(range(demo_num)):
                agentview_images = safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/agentview_rgb"][()].transpose(0, 3, 1, 2))).float()
                eye_in_hand_images = safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/eye_in_hand_rgb"][()].transpose(0, 3, 1, 2))).float()
                joint_states = safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/joint_states"][()])).float()
                gripper_states = safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/gripper_states"][()])).float()
                ee_states = safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/ee_states"][()])).float()
                proprio_states = torch.cat([joint_states, gripper_states, ee_states], dim=1).float()
                proprio = safe_cuda(proprio_states)
                agentview_features = []
                eye_in_hand_features = []

                for j in range(0, len(agentview_images), args.batch_size):
                    batch_images = agentview_images[j:j + args.batch_size].permute(0, 2, 3, 1).cpu().numpy()
                    resized_images = [cv2.resize(img, (448, 448), interpolation=cv2.INTER_NEAREST) for img in batch_images]
                    features = dinov2.process_images(resized_images)
                    agentview_features_batch = rescale_feature_map(torch.as_tensor(features).permute(0, 3, 1, 2), 1, 1, convert_to_numpy=False).squeeze()  # (B, 768)
                    if agentview_features_batch.dim() == 1:
                        agentview_features_batch = agentview_features_batch.unsqueeze(0)
                    agentview_features.append(agentview_features_batch)

                for j in range(0, len(eye_in_hand_images), args.batch_size):
                    batch_images = eye_in_hand_images[j:j + args.batch_size].permute(0, 2, 3, 1).cpu().numpy()
                    resized_images = [cv2.resize(img, (448, 448), interpolation=cv2.INTER_NEAREST) for img in batch_images]
                    features = dinov2.process_images(resized_images)
                    eye_in_hand_features_batch = rescale_feature_map(torch.as_tensor(features).permute(0, 3, 1, 2), 1, 1, convert_to_numpy=False).squeeze()  # (B, 768)
                    if eye_in_hand_features_batch.dim() == 1:
                        eye_in_hand_features_batch = eye_in_hand_features_batch.unsqueeze(0)
                    eye_in_hand_features.append(eye_in_hand_features_batch)

                agentview_features = torch.cat(agentview_features, dim=0)
                eye_in_hand_features = torch.cat(eye_in_hand_features, dim=0)
                embeddings = torch.cat([agentview_features, eye_in_hand_features], dim=1).cpu().unsqueeze(1).numpy().astype('float32')
                if np.isnan(embeddings).any():
                    print("NAN")

                demo_data_grp = grp.create_group(f"demo_{i}")
                demo_data_grp.create_dataset("embedding", data=embeddings)

            # for i in range(demo_num):
            #     agentview_images = safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/agentview_rgb"][()].transpose(0, 3, 1, 2))).float()
            #     eye_in_hand_images = safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/eye_in_hand_rgb"][()].transpose(0, 3, 1, 2))).float()
            #     joint_states = safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/joint_states"][()])).float()
            #     gripper_states = safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/gripper_states"][()])).float()
            #     ee_states = safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/ee_states"][()])).float()
            #     proprio_states = torch.cat([joint_states, gripper_states, ee_states], dim=1).float()

            #     proprio = safe_cuda(proprio_states)
                
            #     agentview_images = agentview_images.permute(0, 2, 3, 1).cpu().numpy()
            #     resized_images = [cv2.resize(img, (448, 448), interpolation=cv2.INTER_NEAREST) for img in agentview_images]
            #     features = dinov2.process_images(resized_images)
            #     agentview_features = rescale_feature_map(features.permute(0, 3, 1, 2), 1, 1, convert_to_numpy=False).squeeze() # (B, 768)


            #     eye_in_hand_images = eye_in_hand_images.permute(0, 2, 3, 1).cpu().numpy()
            #     resized_images = [cv2.resize(img, (448, 448), interpolation=cv2.INTER_NEAREST) for img in eye_in_hand_images]
            #     features = dinov2.process_images(resized_images)
            #     eye_in_hand_features = rescale_feature_map(features.permute(0, 3, 1, 2), 1, 1, convert_to_numpy=False).squeeze() # (B, 768)

            #     embeddings = torch.cat([agentview_features, eye_in_hand_features], dim=1).cpu().unsqueeze(1).numpy().astype('float32')
            #     # if nan in embeddings
            #     if np.isnan(embeddings).any():
            #         print("NAN")
            #         import ipdb; ipdb.set_trace()
            #     # embeddings = torch.cat([agentview_features, eye_in_hand_features, proprio], dim=1).cpu().unsqueeze(1).numpy().astype('float32')

            #     demo_data_grp = grp.create_group(f"demo_{i}")
            #     demo_data_grp.create_dataset("embedding", data=embeddings)

            embedding_file.close()
            f.close()


        

