import argparse
import contextlib
import hashlib
import itertools
import math
import os
import warnings
from pathlib import Path
from typing import Optional
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_depth2img import StableDiffusionDepth2ImgPipeline

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from utils.configs import FinetuneConfig


from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DreamBoothDataset(Dataset):
    def __init__(
        self,
        instance_data_root,
        tokenizer,
        nickname="zxy",
        class_prompt="screw",
        size=512,
        center_crop=False
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.nickname = nickname
        self.class_prompt = class_prompt

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exist.")
        
        # Getting all images inside the specific dataset
        self.instance_images_path = list(self.instance_data_root.rglob("*.jpg")) + list(self.instance_data_root.rglob("*.png"))
        
        # transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.instance_images_path)

    def __getitem__(self, index):
        example = {}
        
        # 1. 이미지 로드 및 텐서 변환
        image_path = self.instance_images_path[index]
        instance_image = Image.open(image_path)
        
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
            
        example["instance_images"] = self.image_transforms(instance_image)
        
        # 🌟 마법 2: 이미지가 들어있던 '폴더 이름'을 추출! (예: "vanilla", "horizontal")
        folder_name = image_path.parent.name
        
        # 🌟 마법 3: 폴더 이름에 따라 캡틴이 원하던 프롬프트를 찰떡같이 부여!
        if folder_name == "horizontal":
            prompt = f"A horizontal cross-section of a {self.nickname} {self.class_prompt}"
        elif folder_name == "vertical":
            prompt = f"A vertical cross-section of a {self.nickname} {self.class_prompt}"
        else: # "vanilla" 이거나 최상위 폴더에 있는 경우
            prompt = f"A {self.nickname} {self.class_prompt}"
            
        # 4. 텍스트를 숫자로 변환 (토크나이징)
        example["instance_prompt_ids"] = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return example
    


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples
        print(f'Creating prompt dataset with prompt={prompt} and num_samples={num_samples}')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


    
class DreamBooth:
    def __init__(self, args: FinetuneConfig):
        self.args = args
        
        if args.seed is not None:
            set_seed(args.seed)
        
        if args.output_dir is not None:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            
        logging_dir = Path(args.output_dir, args.logging_dir)
        accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
        
        # getting the accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with="wandb",
            project_config=accelerator_project_config,
        )
        
        # if there is problem in accelerating
        if args.train_text_encoder and args.gradient_accumulation_steps > 1 and self.accelerator.num_processes > 1:
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )
            
        # class image should be generated
        if args.with_prior_preservation:
            self._prepare_class_images()
        
        # getting tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name or args.pretrained_model_name_or_path,
            revision=args.revision,
            use_fast=False,
        )
        
        self.text_encoder_cls = self.import_model_class_from_model_name_or_path()
        
        # Load models and create wrapper for stable diffusion
        self.text_encoder = self.text_encoder_cls.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
        )
        if not args.train_text_encoder:
            text_encoder.requires_grad_(False)
        
        # getting variational autoencoder
        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
        )
        self.vae.requires_grad_(False)
        
        # getting U-Net
        self.unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.revision,
        )
        
        if args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if args.train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()
        
        if is_xformers_available():
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                warnings.warn(
                    "Could not enable memory efficient attention. Make sure xformers is installed"
                    f" correctly and a GPU is available: {e}"
                )

        if args.scale_lr:
            self.args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * self.accelerator.num_processes
            )
            
        
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
            
            optimizer_class = torch.optim.AdamW 

        params_to_optimize = (
            itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
        )
        
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                self.optimizer_class = bnb.optim.AdamW8bit
            except ImportError:
                raise ImportError("Install the library to use 8-bit Adam: `pip install bitsandbytes`")
        else:
            self.optimizer_class = torch.optim.AdamW
        
        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

        vae_scale_factor = create_depth_images([args.instance_data_dir, args.class_data_dir], args.pretrained_model_name_or_path, accelerator, unet, text_encoder, revision = args.revision)
        train_dataset = DreamBoothDataset(
            instance_data_root=args.instance_data_dir,
            instance_prompt=args.instance_prompt,
            tokenizer=tokenizer,
            vae_scale_factor=vae_scale_factor,
            class_data_root=args.class_data_dir if args.with_prior_preservation else None,
            class_prompt=args.class_prompt,
            size=args.resolution,
            center_crop=args.center_crop,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
            num_workers=1,
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

        if args.train_text_encoder:
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )
        accelerator.register_for_checkpointing(lr_scheduler)

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move text_encode and vae to gpu.
        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        vae.to(accelerator.device, dtype=weight_dtype)
        if not args.train_text_encoder:
            text_encoder.to(accelerator.device, dtype=weight_dtype)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers("dreambooth", config=vars(args))
        
        self.accelerator.init_trackers(
            project_name="ddujjonku-dreambooth",
            config=vars(self.args)               
        )
                    
    # ==========================================================
    # 🛠️ Helper Methods (도우미 함수들)
    # ==========================================================
    
    def import_model_class_from_model_name_or_path(self):
        text_encoder_config = PretrainedConfig.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.args.revision,
        )
        model_class = text_encoder_config.architectures[0]

        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel
            return CLIPTextModel
        elif model_class == "RobertaSeriesModelWithTransformation":
            from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
            return RobertaSeriesModelWithTransformation
        else:
            raise ValueError(f"{model_class} is not supported.")
        
    
    
    def _prepare_class_images(self):
        """
        Prior Preservation을 위한 클래스 이미지 자동 생성기!
        __init__이 너무 뚱뚱해지는 걸 막기 위해 밖으로 빼냈어!
        """
        class_images_dir = Path(self.args.class_data_dir)
        class_images_dir.mkdir(parents=True, exist_ok=True) # 우아한 pathlib 사용
        
        # 현재 생성된 이미지 개수 파악
        cur_class_images = len(list(class_images_dir.glob("*.jpg")) + list(class_images_dir.glob("*.png")))

        # 이미 충분히 만들어져 있다면 쿨하게 종료!
        if cur_class_images >= self.args.num_class_images:
            return 

        # 부족한 만큼 새로 생성 시작!
        num_new_images = self.args.num_class_images - cur_class_images
        print(f"Number of class images to sample: {num_new_images}.")

        torch_dtype = torch.float16 if self.accelerator.device.type == "cuda" else torch.float32
        pipeline = DiffusionPipeline.from_pretrained(
            self.args.pretrained_txt2img_model_name_or_path, # 주의: 보통 Base 모델 경로를 넣음
            torch_dtype=torch_dtype,
            safety_checker=None,
            revision=self.args.revision,
        )
        pipeline.set_progress_bar_config(disable=True)
        pipeline.to(self.accelerator.device)

        # PromptDataset (기존 코드에 있던 커스텀 클래스)
        sample_dataset = PromptDataset(self.args.class_prompt, num_new_images)
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=self.args.sample_batch_size)
        sample_dataloader = self.accelerator.prepare(sample_dataloader)

        for example in tqdm(
            sample_dataloader, desc="Generating class images", disable=not self.accelerator.is_local_main_process
        ):
            images = pipeline(example["prompt"]).images

            for i, image in enumerate(images):
                hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                image.save(image_filename)

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()