


import os
import torch
import CLIP.clip as clip
# import clip
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from captum.attr import visualization
import random
from CLIP.simple_tokenizer import SimpleTokenizer as _Tokenizer
import cv2
from PIL import Image
import random

import json
from transformers import BlipProcessor, BlipForConditionalGeneration

# start_layer =  -1
# start_layer_text =  -1

# _tokenizer = _Tokenizer()

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

clip._MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

class ImageProcessor:
    def __init__(self, start_layer=-1, start_layer_text=-1, _tokenizer=_Tokenizer(), cls_name=None):
        self.start_layer = start_layer
        self.start_layer_text = start_layer_text
        self._tokenizer = _tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.cls_name = cls_name



    def interpret(self, image, texts, model, device):
        batch_size = texts.shape[0]
        images = image.repeat(batch_size, 1, 1, 1)
        logits_per_image, logits_per_text = model(images, texts)
        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
        index = [i for i in range(batch_size)]
        one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
        one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * logits_per_image)
        model.zero_grad()

        image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

        if self.start_layer == -1: 
        # calculate index of last lareyer 
            start_layer = len(image_attn_blocks) - 1
        
        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
        R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        for i, blk in enumerate(image_attn_blocks):
            if i < start_layer:
                continue
            grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
            cam = blk.attn_probs.detach()
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R = R + torch.bmm(cam, R)
        image_relevance = R[:, 0, 1:]

        
        text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

        if self.start_layer_text == -1: 
        # calculate index of last layer 
            start_layer_text = len(text_attn_blocks) - 1

        num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
        R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
        R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        for i, blk in enumerate(text_attn_blocks):
            if i < start_layer_text:
                continue
            grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
            cam = blk.attn_probs.detach()
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R_text = R_text + torch.bmm(cam, R_text)
        text_relevance = R_text
    
        return text_relevance, image_relevance
    
    def show_image_relevance(self, image_relevance, image, orig_image):
        # create heatmap from mask on image
        def show_cam_on_image(img, mask):
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam = heatmap + np.float32(img)
            cam = cam / np.max(cam)
            return cam

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(orig_image);
        axs[0].axis('off');

        dim = int(image_relevance.numel() ** 0.5)
        image_relevance = image_relevance.reshape(1, 1, dim, dim)
        image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
        image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
        image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
        image = image[0].permute(1, 2, 0).data.cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())
        vis = show_cam_on_image(image, image_relevance)
        vis = np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        axs[1].imshow(vis);
        axs[1].axis('off');
        

    def show_heatmap_on_text(self, text, text_encoding, R_text):
        CLS_idx = text_encoding.argmax(dim=-1)
        R_text = R_text[CLS_idx, 1:CLS_idx]
        text_scores = R_text / R_text.sum()
        text_scores = text_scores.flatten()
        print(text_scores)
        text_tokens=self._tokenizer.encode(text)
        text_tokens_decoded=[self._tokenizer.decode([a]) for a in text_tokens]
        vis_data_records = [visualization.VisualizationDataRecord(text_scores,0,0,0,0,0,text_tokens_decoded,1)]
        visualization.visualize_text(vis_data_records)
    




    
    
    def emphasize_attention(self, image_relevance, image):
        # Create heatmap from attention mask
        def create_heatmap(mask):
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            return heatmap

        # Apply attention-based augmentation
        dim = int(image_relevance.numel() ** 0.5)
        image_relevance = image_relevance.reshape(1, 1, dim, dim)
        image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
        image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
        image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())


        max_relevance = np.max(image_relevance)
        max_indices = np.argwhere(image_relevance == max_relevance)
        max_score = [(idx[1], idx[0]) for idx in max_indices]

        unique_values, value_counts = np.unique(image_relevance, return_counts=True)
        frequency_dict = dict(zip(unique_values, value_counts)) # Attention 값 빈도수

        # Emphasize attention regions in the image
        image = image[0].permute(1, 2, 0).data.cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())

        # Apply attention mask to image
        heatmap = create_heatmap(image_relevance)
        
        emphasized_image = (heatmap * image) + ((1 - heatmap) * 255)
        emphasized_image = np.uint8(emphasized_image)

        return emphasized_image, max_score


    def crop_object_attention(self, image, max_score):
        # Randomly select a pixel from max_score
        selected_pixel = random.choice(max_score)
        

        # Set the region size for cropping the object
        min_region_size = int(0.50 * image.shape[0])  # 50% of image height
        max_region_size = int(0.80 * image.shape[0])  # 80% of image height
        

        # Randomly generate the region size within the specified range
        region_size = random.randint(min_region_size, max_region_size)

        # Extract the coordinates of the selected pixel
        x, y = selected_pixel

        # Define the coordinates for the region to be cropped
        x_start = max(0, x - region_size // 2)
        x_end = min(image.shape[1], x + region_size // 2)
        y_start = max(0, y - region_size // 2)
        y_end = min(image.shape[0], y + region_size // 2)

        # Crop the region from the image
        cropped_image = image[y_start:y_end, x_start:x_end].copy()

        return cropped_image
    
    def generate_image(self):
            
        # 경로 설정
        base_path = f"./data/cosmetic/images/train/{self.cls_name}"

        # 경로 내의 모든 이미지 파일 가져오기
        image_files = []
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith(".jpg"):
                    image_files.append(os.path.join(root, file))

        # 이미지별로 처리
        for img_path in image_files:
            # 파일 경로를 '/'로 분할하여 경로 요소를 얻습니다
            path_elements = img_path.split(os.path.sep)
            # 카테고리는 폴더 구조에서 두 번째-to-last 요소로 가정합니다 (예: "Abacus 5")
            category = path_elements[-2]
            print(category)
            img = self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
            
            # "a photo of {카테고리}" 형식의 텍스트 생성
            text = clip.tokenize([f"a photo of {category}"]).to(self.device)

            logits_per_image, logits_per_text = self.model(img, text)
            # print(color.BOLD + color.PURPLE + color.UNDERLINE + f'CLIP similarity score: {logits_per_image.item()}' + color.END)

            R_text, R_image = self.interpret(model=self.model, image=img, texts=text, device=self.device)
            batch_size = text.shape[0]
            for i in range(batch_size):
                _, max_score = self.emphasize_attention(R_image[i], img)

            # 원본 이미지 로드
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 이미지 크기 변경
            target_size = (224, 224)
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

            # 이미지 크롭
            cropped_image = self.crop_object_attention(image, max_score)

            # 크롭된 이미지를 원본 이미지의 위치에 저장
            cropped_img_path = img_path.replace(".jpg", "_cropped.jpg")
            Image.fromarray(cropped_image).save(cropped_img_path)
            
    def generate_prompt(self):
        # 이미지 파일들이 있는 폴더의 경로를 설정
        image_folder = f"./data/cosmetic/images/train/{self.cls_name}"

        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        model.to(self.device)

        result_dict = {}

        for class_folder in os.listdir(image_folder):
            class_folder_path = os.path.join(image_folder, class_folder)
            
            if os.path.isdir(class_folder_path):
                class_name = class_folder  # 클래스 이름

                if class_name not in result_dict:
                    result_dict[class_name] = []

                for image_filename in os.listdir(class_folder_path):
                    image_path = os.path.join(class_folder_path, image_filename)

                    # 이미지 열기
                    raw_image = Image.open(image_path)

                    # 이미지 캡션 생성
                    inputs = processor(raw_image, return_tensors="pt")
                    inputs.to(self.device)
                    out = model.generate(**inputs)
                    caption = processor.decode(out[0], skip_special_tokens=True)

                    # 이미지 캡션을 결과 딕셔너리에 추가
                    result_dict[class_name].append(caption)

        # JSON 파일에 결과 저장
        output_json_path = f'./gpt3_prompts/CuPL_prompts_{self.cls_name}.json'

        # JSON 파일 생성 및 데이터 쓰기
        with open(output_json_path, 'w') as json_file:
            json.dump(result_dict, json_file, indent=4)
            
        

        for class_folder in os.listdir(image_folder):
            class_folder_path = os.path.join(image_folder, class_folder)

            if os.path.isdir(class_folder_path):
                # 클래스 폴더 내의 이미지 파일 순회
                for image_filename in os.listdir(class_folder_path):
                    image_path = os.path.join(class_folder_path, image_filename)

                    # 파일 이름에 'crop' 키워드가 포함되어 있으면 삭제
                    if 'crop' in image_filename:
                        os.remove(image_path)
                        print(f"Deleted: {image_filename}")

        
        
                

# Inference
if __name__ == '__main__':
  ImageProcessor = ImageProcessor(cls_name = 'Dior')
  ImageProcessor.generate_image()
  ImageProcessor.generate_prompt()