import io
import os
import sys
import PIL
import cv2
import torch
import einops
import base64
import requests
import subprocess
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from diffusers import ControlNetModel
from diffusers.utils import load_image
from torchvision.utils import save_image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from diffusers import StableDiffusionInpaintPipeline, UniPCMultistepScheduler
# functions from intern files:
models_path = Path("..") / "models"
sys.path.append(str(models_path.resolve()))
from utils.stable_diffusion_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline
from annotator.util import resize_image, HWC3
sys.path.remove(str(models_path.resolve()))

#new

# end-new





class Item(BaseModel):
    prompt: Optional[str]
    height: Optional[int] = 512
    width: Optional[int] = 512
    num_inference_steps: Optional[int] = 25
    num_images_per_prompt: Optional[int] = 1







device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "stabilityai/stable-diffusion-2-inpainting"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16,)
# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed
pipe.enable_xformers_memory_efficient_attention()
pipe = pipe.to("cuda")







def download_image(url):
    response = requests.get(url)
    img_bytes = BytesIO(response.content)
    img = PIL.Image.open(img_bytes).convert("RGB")
    return np.asarray(img)

img_url = "https://uptechai.com/production-images/image.png"
mask_url = "https://uptechai.com/production-images/mask.png"






def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    full_img = None

    # for ann in sorted_anns:
    for i in range(len(sorted_anns)):
        ann = anns[i]
        m = ann['segmentation']
        if full_img is None:
            full_img = np.zeros((m.shape[0], m.shape[1], 3))
            map = np.zeros((m.shape[0], m.shape[1]), dtype=np.uint16)
        map[m != 0] = i + 1
        color_mask = np.random.random((1, 3)).tolist()[0]
        full_img[m != 0] = color_mask
    full_img = full_img*255
    # anno encoding from https://github.com/LUSSeg/ImageNet-S
    res = np.zeros((map.shape[0], map.shape[1], 3))
    res[:, :, 0] = map % 256
    res[:, :, 1] = map // 256
    res.astype(np.float32)
    full_img = Image.fromarray(np.uint8(full_img))
    return full_img, res



def get_sam_control(image):
    #
    sam_checkpoint = "models/sam_vit_h_4b8939.pth"
    model_type = "default"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    #
    masks = mask_generator.generate(image)
    full_img, res = show_anns(masks)
    return full_img, res





def process(source_image, enable_all_generate, mask_image, control_scale, enable_auto_prompt, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale):
    sed = 232342
    eta = 0.0
    input_image = source_image["image"]
    if mask_image is None:
        if enable_all_generate:
            print("source_image", source_image["mask"].shape, input_image.shape,)
            print(source_image["mask"].max())
            mask_image = np.ones((input_image.shape[0], input_image.shape[1], 3))*255
        else:
            mask_image = source_image["mask"]


    with torch.no_grad():
        input_image = HWC3(input_image)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        # logger.info("Generating SAM seg:")
        # the default SAM model is trained with 1024 size.
        full_segmask, detected_map = get_sam_control(
            resize_image(input_image, detect_resolution))

        detected_map = HWC3(detected_map.astype(np.uint8))
        detected_map = cv2.resize(
            detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(
            detected_map.copy()).float().cuda()
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        mask_image = HWC3(mask_image.astype(np.uint8))
        mask_image = cv2.resize(
            mask_image, (W, H), interpolation=cv2.INTER_LINEAR)
        mask_image = Image.fromarray(mask_image)


        # if seed == -1:
        #     seed = random.randint(0, 65535)
        # seed_everything(seed)
        generator = torch.manual_seed(seed)
        
        x_samples = pipe(
            image=img,
            mask_image=mask_image,
            prompt=[prompt + ', ' + a_prompt] * num_samples,
            negative_prompt=[n_prompt] * num_samples,  
            num_images_per_prompt=num_samples,
            num_inference_steps=ddim_steps, 
            generator=generator, 
            height=H,
            width=W,
        ).images


        results = [x_samples[i] for i in range(num_samples)]
    return [full_segmask, mask_image] + results, prompt







# disable gradio when not using GUI.
def prepare_process(img, mask, logger):
    enable_auto_prompt = True
    # input_image = np.array(input_image, dtype=np.uint8)
    # mask_image = np.array(mask_image, dtype=np.uint8)
    input_image = img
    mask_image = mask
    prompt = "on top of a rock at the beach"
    a_prompt = 'best quality, extremely detailed'
    n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
    num_samples = 3
    image_resolution = 512
    detect_resolution = 512
    ddim_steps = 30
    guess_mode = False
    strength = 1.0
    scale = 9.0
    seed = 23482034805
    eta = 0.0

    outputs = process(input_image, mask_image, enable_auto_prompt, prompt, a_prompt, n_prompt, num_samples, image_resolution,
                    detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)

    image_list = []
    input_image = resize_image(input_image, 512)
    image_list.append(torch.tensor(input_image))
    for i in range(len(outputs)):
        each = outputs[i]
        if type(each) is not np.ndarray:
            each = np.array(each, dtype=np.uint8)
        each = resize_image(each, 512)
        image_list.append(torch.tensor(each))

    image_list = torch.stack(image_list).permute(0, 3, 1, 2)

    buffered = io.BytesIO()
    save_image(image_list, buffered, nrow=3,
               normalize=True, value_range=(0, 255))
    buffered.seek(0)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str










#installs all packages into models folder
def extra_packages(logger):
    # putting required packages into the models folder
    def install_package(package_name, repo_url, logger):
        folder_path = "./models"
        package_path = os.path.join(folder_path, package_name)
        if not os.path.exists(package_path):
            # clone the repository to the folder_path
            subprocess.run(["git", "clone", repo_url, package_path])
            
            # install the package from the local folder
            subprocess.run(["pip", "install", "-e", package_path]) 
            logger.info(f"{package_name} installed...")
        else:
            logger.info(f"{package_name} already exists. Skipping installation.")

    # install packages
    install_package("CLIP", "https://github.com/openai/CLIP.git", logger)
    install_package("detectron2", "https://github.com/facebookresearch/detectron2.git", logger)
    install_package("GroundingDINO", "https://github.com/IDEA-Research/GroundingDINO.git", logger)
#










def predict(item, run_id, logger):
    item = Item(**item)  
    img = download_image(img_url)
    mask = download_image(mask_url)

    #
    extra_packages(logger) 
    #
    #download models into the modules folder
    files_to_download = [
        {
            "url": "https://github.com/Cheems-Seminar/segment-anything-and-name-it/releases/download/v1.0/swinbase_part_0a0000.pth",
            "file_path": "./models/swinbase_part_0a0000.pth",
            "file_name": "swinbase_part_0a0000",
        },
        {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "file_path": "./models/sam_vit_h_4b8939.pth",
            "file_name": "sam_vit_h_4b8939",
        },
        {
            "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth",
            "file_path": "./models/groundingdino_swinb_cogcoor.pth",
            "file_name": "groundingdino_swinb_cogcoor",
        },
    ]

    for file in files_to_download:
        url = file["url"]
        file_path = file["file_path"]
        file_name = file["file_name"]

        if not os.path.exists(file_path):
            logger.info(f'Downloading {file_name} file...')
            response = requests.get(url)
            with open(file_path, "wb") as f:
                f.write(response.content)
            logger.info('Download complete')
        else:
            logger.info(f'File {file_name} already exists')
    #
    output = prepare_process(img, mask, logger)
    #

    return output























    # images = pipe(
    #     prompt=item.prompt,
    #     height=item.height,
    #     width=item.width,
    #     image=img,
    #     mask_image=mask,
    #     num_images_per_prompt=item.num_images_per_prompt,
    #     num_inference_steps=item.num_inference_steps
    # ).images
    # logger.info('not here')
    # finished_images = []
    # for image in images:
    #     buffered = io.BytesIO()
    #     image.save(buffered, format="PNG")
    #     finished_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

    # return finished_images