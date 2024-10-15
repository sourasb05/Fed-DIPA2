import warnings
warnings.filterwarnings("ignore")

import torchvision
import torch
import torch.nn as nn
from torchvision import transforms as T

import cv2
import numpy as np
import os
import shutil
from glob import glob
import tqdm
import sys

from PIL import Image
import torchvision.transforms.functional as TF

import pandas as pd
import json
import clip

from torchvision.ops import roi_align

class FeatureGenerator():
    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ## "openai_ViT-L/14@336px" , "resnet50"
        self.org_model_name = "openai_ViT-L/14@336px"
        # self.org_model_name = "resnet50"
        self.model_name = self.org_model_name.replace("/", "-")
        self.model = self.get_model()

        self.images_dir = "./dipa2/images/"
        self.features_dir = "image_features/"
        self.object_features_dir = "object_features/"
        self.annotation_file = "./dipa2/annotations_filtered_bbox.csv"
        self.image_size = (224, 224)
        self.padding_color = (0, 0, 0)
        self.max_bboxes = 32
        self.roi_sampling_ratio = 1

    def get_model(self):
        model = None
        if self.model_name == "resnet50":       
            resnet = torchvision.models.resnet50(pretrained=True)
            model = nn.Sequential(*list(resnet.children())[:-2])
            model.eval()
            model.to(self.device)

            self.avg_pool = nn.Sequential(list(resnet.children())[-2])
            self.avg_pool.eval()

        elif "openai" in self.model_name:
            model = clip.load(self.org_model_name.split("_")[1], device=self.device)

        return model

    def generate_image_features(self, image_path):
        features = None

        if self.model_name == "resnet50":
            features = self.generate_image_features_cnn(image_path)
        elif "openai" in self.model_name:
            features = self.generate_image_features_clip(image_path)

        return features
    
    def generate_image_features_clip(self, image_or_path, is_image=False):
        clip_model, preprocess = self.model

        image_data = image_or_path
        if not is_image:
            image_data = Image.open(image_or_path)

        image = preprocess(image_data).unsqueeze(0).to(self.device)

        features = None
        with torch.no_grad():
            img_emb = clip_model.encode_image(image)[0]
            features = torch.tensor(img_emb)  
        return features

    def generate_image_features_cnn(self, image_path):
        image = Image.open(image_path).convert('RGB')
        w, h = image.size
        ratio = min(self.image_size[0] / h, self.image_size[1] / w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        
        image = TF.resize(image, (new_h, new_w))
        image = TF.pad(image, padding=(0, 0, self.image_size[1] - new_w, 
                                       self.image_size[0] - new_h), fill=self.padding_color)
        image = TF.to_tensor(image)

        transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
        image = transform(image)

        th, tw = self.image_size 
        data_dim = (1, 3, th, tw)
        data = torch.full(data_dim, 0)
        
        data[0] = image
        data = data.float()
        data = data.to(self.device)
        with torch.no_grad(): 
            features = self.model(data)
        return features[0]

    def generate_all_image_features(self):
        model_features_dir = os.path.join(self.features_dir, self.model_name)
        if os.path.exists(model_features_dir):
            shutil.rmtree(model_features_dir)
        os.makedirs(model_features_dir)

        image_paths = glob(self.images_dir + "/*")
        for image_path in tqdm.tqdm(image_paths, total=len(image_paths),
                                    desc="Generating Image Features"):

            image_name = os.path.splitext(os.path.basename(image_path))[0]
            features_path = os.path.join(model_features_dir, image_name + ".pt")

            features = self.generate_image_features(image_path)
            torch.save(features.clone().detach().cpu(), features_path)

    def generate_object_features(self):

        self.generate_all_image_features()     

        mega_table = pd.read_csv(self.annotation_file)

        model_features_dir = os.path.join(self.object_features_dir, self.model_name)
        if os.path.exists(model_features_dir):
            shutil.rmtree(model_features_dir)
        os.makedirs(model_features_dir)

        ignored_sample = 0

        for i, row in tqdm.tqdm(mega_table.iterrows(), total = len(mega_table), desc="Generating Object Features"):
            image_basename = row["imagePath"]
            unique_object_id = row["ObjectAnnotatorId"]

            image_name = os.path.splitext(image_basename)[0]

            image_height = row['height']
            image_width = row['width']

            ratio = min(self.image_size[0] / image_height, self.image_size[1] / image_width)
        
            bboxes_original = json.loads(row['bbox'])
            bboxes = []

            for ib, bbox in enumerate(bboxes_original):
                x, y, w, h = bbox

                x *= ratio
                y *= ratio
                w *= ratio
                h *= ratio

                x1, y1, x2, y2 = x, y, x+w, y+h

                if x1 < 0: x1 = 0
                if x2 >= self.image_size[0] : x2 = self.image_size[0]-1
                if y1 < 0: y1 = 0
                if y2 >= self.image_size[1] : y2 = self.image_size[1]-1

                if x1 >= x2 or y1 >= y2:
                    #print("improper bbox - ", [x1, y1, x2, y2])
                    continue

                bboxes.append(list(map(int, [0, x1, y1, x2, y2])))

            num_bboxes = len(bboxes)

            if num_bboxes == 0:
                ignored_sample += 1
                continue

            if num_bboxes > self.max_bboxes -1:
                print("number of bboxes limit exceeded - %d vs %d (max-limit) " % (num_bboxes, self.max_bboxes-1))
                sys.exit(1)

            image_features_path =  os.path.join(self.features_dir, self.model_name, image_name + ".pt")
            image_features = torch.load(image_features_path).to(self.device)

            object_features_path = os.path.join(model_features_dir, unique_object_id + ".pt")
            bb_features = torch.zeros((self.max_bboxes, 
                                    image_features.shape[0]))
            
            if self.model_name == "resnet50":

                roi_output_dim = tuple(image_features.shape[1:])
                roi_spatial_scale = image_features.shape[-1]/float(self.image_size[0])

                with torch.no_grad():

                    bb_features[0] = self.avg_pool(image_features).flatten()
                    roi_image_features = torch.stack([image_features.clone()])
                    bboxes = torch.tensor(bboxes).float().to(self.device)

                    for ib, bbox in enumerate(bboxes):
                        roi_bbox = torch.stack([bbox])
                        ra_features = roi_align(roi_image_features,
                                                roi_bbox,
                                                roi_output_dim,
                                                roi_spatial_scale,
                                                sampling_ratio=self.roi_sampling_ratio,
                                                aligned=True)

                        ra_features = self.avg_pool(ra_features).flatten(start_dim=1)
                        bb_features[ib+1] = ra_features[0]

            elif "openai" in self.model_name:
                image_path = os.path.join(self.images_dir, image_name + ".jpg")
                image = Image.open(image_path)

                with torch.no_grad():
                    bb_features[0] = image_features
                    count = 0
                    for ib, bbox in enumerate(bboxes):
                        _, x1, y1, x2, y2 = bbox

                        width = x2-x1
                        height = y2-y1

                        if width < 5 or height < 5:
                            continue

                        crop = image.crop((x1, y1, x2, y2))

                        crop_features = self.generate_image_features_clip(crop, is_image=True)
                        bb_features[count+1] = crop_features
                        count += 1

            torch.save(bb_features.clone().detach().cpu(), object_features_path)
        #print("#ignored_samples", ignored_sample)

if __name__ == "__main__":
   feature_generator = FeatureGenerator()
   feature_generator.generate_object_features()
