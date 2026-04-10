import os
import math
import hashlib
import itertools
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import AutoTokenizer, PretrainedConfig

from utils.configs import FinetuneConfig

logger = get_logger(__name__)

# ==========================================================
# 🛠️ Helper Functions & Datasets
# ==========================================================

def get_depth_image_path(normal_image_path):
    """원본 이미지 경로를 받아서, 동일한 폴더에 _depth.png를 붙여서 반환합니다."""
    return normal_image_path.parent / f"{normal_image_path.stem}_depth.png"

class PromptDataset(Dataset):
    """다중 GPU 환경에서 Class Image 생성을 돕는 착한 도우미 데이터셋"""
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return {"prompt": self.prompt, "index": index}

class DreamBoothDataset(Dataset):
    """캡틴의 다중 뷰(Multi-view) 로직과 Depth 로직이 완벽하게 융합된 궁극의 데이터셋!"""
    def __init__(
        self,
        instance_data_root,
        tokenizer,
        vae_scale_factor,
        nickname="zxy",
        class_prompt="screw",
        class_data_root=None,
        size=512,
        center_crop=False
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.vae_scale_factor = vae_scale_factor
        self.nickname = nickname
        self.class_prompt_text = class_prompt

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exist.")
        
        # 🌟 꿀팁: rglob으로 모든 하위 폴더 이미지를 긁어오되, _depth가 붙은 건 제외!
        all_files = list(self.instance_data_root.rglob("*.jpg")) + list(self.instance_data_root.rglob("*.png"))
        self.instance_images_path = [p for p in all_files if "_depth." not in p.name]
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        # Class 데이터 세팅
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            all_class_files = list(self.class_data_root.rglob("*.jpg")) + list(self.class_data_root.rglob("*.png"))
            self.class_images_path = [p for p in all_class_files if "_depth." not in p.name]
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        self.depth_image_transforms = transforms.Compose([
            transforms.Resize(size // self.vae_scale_factor, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        
        # 1. Instance (캡틴의 다중 뷰 로직 + Depth)
        instance_image_path = self.instance_images_path[index % self.num_instance_images]
        instance_depth_image_path = get_depth_image_path(instance_image_path)
        
        instance_image = Image.open(instance_image_path).convert("RGB")
        instance_depth_image = Image.open(instance_depth_image_path)
        
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_depth_images"] = self.depth_image_transforms(instance_depth_image)
        
        # 폴더 이름 기반 스마트 프롬프팅
        folder_name = instance_image_path.parent.name
        if folder_name == "horizontal":
            prompt = f"A horizontal cross-section of a {self.nickname} {self.class_prompt_text}"
        elif folder_name == "vertical":
            prompt = f"A vertical cross-section of a {self.nickname} {self.class_prompt_text}"
        else:
            prompt = f"A {self.nickname} {self.class_prompt_text}"
            
        example["instance_prompt_ids"] = self.tokenizer(
            prompt, truncation=True, padding="max_length",
            max_length=self.tokenizer.model_max_length, return_tensors="pt"
        ).input_ids

        # 2. Class Image 처리 (Prior Preservation)
        if self.class_data_root:
            class_image_path = self.class_images_path[index % self.num_class_images]
            class_depth_image_path = get_depth_image_path(class_image_path)
            
            class_image = Image.open(class_image_path).convert("RGB")
            class_depth_image = Image.open(class_depth_image_path)
            
            example["class_images"] = self.image_transforms(class_image)
            example["class_depth_images"] = self.depth_image_transforms(class_depth_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt_text, truncation=True, padding="max_length",
                max_length=self.tokenizer.model_max_length, return_tensors="pt"
            ).input_ids

        return example

def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    depth_values = [example["instance_depth_images"] for example in examples]

    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        depth_values += [example["class_depth_images"] for example in examples]

    pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
    depth_values = torch.stack(depth_values).to(memory_format=torch.contiguous_format).float()
    input_ids = torch.cat(input_ids, dim=0)

    return {"input_ids": input_ids, "pixel_values": pixel_values, "depth_values": depth_values}


# ==========================================================
# 🚀 The Ultimate DreamBooth Class
# ==========================================================

class DreamBooth:
    def __init__(self, args: FinetuneConfig):
        """1단계: 작업실 세팅 및 부품 가져오기"""
        self.args = args
        
        if args.seed is not None:
            set_seed(args.seed)
        
        if args.output_dir is not None:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            
        accelerator_project_config = ProjectConfiguration(
            project_dir=args.output_dir, 
            logging_dir=Path(args.output_dir, args.logging_dir)
        )
        
        # 🌟 깐지나는 W&B 세팅 적용 완료! Seonhana 등판!
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with="wandb",
            project_config=accelerator_project_config,
        )
        
        if args.train_text_encoder and args.gradient_accumulation_steps > 1 and self.accelerator.num_processes > 1:
            raise ValueError("Gradient accumulation is not supported for text encoder in distributed training.")
        
        # 토크나이저 및 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name or args.pretrained_model_name_or_path,
            revision=args.revision, use_fast=False,
        )
        
        self.text_encoder_cls = self._import_model_class()
        self.text_encoder = self.text_encoder_cls.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision,
        )
        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision,
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision,
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )

        # 메모리 최적화 세팅
        self.vae.requires_grad_(False)
        if not args.train_text_encoder:
            self.text_encoder.requires_grad_(False)
            
        if args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if args.train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()
                
        if is_xformers_available():
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning(f"Could not enable xformers: {e}")

    # ========================================
    # ⚙️ Data Preparation Methods
    # ========================================

    def prepare_data_and_optimizers(self):
        """2단계: 데이터셋, Depth 이미지, 옵티마이저 생성 및 Accelerator 연결"""
        
        # 1. Class Image 생성 & Depth Image 생성
        if self.args.with_prior_preservation:
            self._generate_class_images()
            
        paths_to_process = [self.args.instance_data_dir]
        if self.args.with_prior_preservation:
            paths_to_process.append(self.args.class_data_dir)
            
        vae_scale_factor = self._create_depth_images(paths_to_process)

        # 2. Dataset & DataLoader
        self.train_dataset = DreamBoothDataset(
            instance_data_root=self.args.instance_data_dir,
            instance_prompt=self.args.instance_prompt, # Legacy 파라미터 (내부에서 오버라이드 됨)
            tokenizer=self.tokenizer,
            vae_scale_factor=vae_scale_factor,
            class_data_root=self.args.class_data_dir if self.args.with_prior_preservation else None,
            class_prompt=self.args.class_prompt,
            size=self.args.resolution,
            center_crop=self.args.center_crop,
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, self.args.with_prior_preservation),
            num_workers=1,
        )

        # 3. 옵티마이저 설정 (8bit 버그 완벽 수정!)
        if self.args.scale_lr:
            self.args.learning_rate *= (self.args.gradient_accumulation_steps * self.args.train_batch_size * self.accelerator.num_processes)

        if self.args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_class = bnb.optim.AdamW8bit
            except ImportError:
                raise ImportError("`pip install bitsandbytes`를 설치하세요!")
        else:
            optimizer_class = torch.optim.AdamW

        params_to_optimize = itertools.chain(self.unet.parameters(), self.text_encoder.parameters()) if self.args.train_text_encoder else self.unet.parameters()
        
        self.optimizer = optimizer_class(
            params_to_optimize, lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay, eps=self.args.adam_epsilon,
        )

        # 4. 스케줄러 & Accelerator Prepare
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler, optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps * self.args.gradient_accumulation_steps,
        )

        if self.args.train_text_encoder:
            self.unet, self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
                self.unet, self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler
            )
        else:
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
                self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
            )
            
        self.accelerator.register_for_checkpointing(self.lr_scheduler)

        # VAE 정밀도 설정 및 GPU 이동
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16": weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16": weight_dtype = torch.bfloat16

        self.vae.to(self.accelerator.device, dtype=weight_dtype)
        if not self.args.train_text_encoder:
            self.text_encoder.to(self.accelerator.device, dtype=weight_dtype)
            
        self.weight_dtype = weight_dtype

        # 🌟 W&B 로깅 시작!
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(project_name="ddujjonku-dreambooth", config=vars(self.args))


    # ========================================
    # 🏃‍♂️ Training Loop
    # ========================================

    def train(self):
        """3단계: 폭풍의 훈련 루프!"""
        logger.info("***** Running training *****")
        global_step = 0
        first_epoch = 0

        progress_bar = tqdm(range(global_step, self.args.max_train_steps), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.unet.train()
            if self.args.train_text_encoder:
                self.text_encoder.train()
                
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    # 1. Latent 생성
                    latents = self.vae.encode(batch["pixel_values"].to(dtype=self.weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215

                    # 2. 노이즈 추가
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                    
                    # 🌟 3. Depth Map 연결 (핵심!)
                    noisy_latents = torch.cat([noisy_latents, batch["depth_values"]], dim=1)

                    # 4. 예측 및 Loss 계산
                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
                    model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    target = noise # epsilon prediction 가정

                    if self.args.with_prior_preservation:
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                        loss = loss + self.args.prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # 5. Backprop & Update
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        params_to_clip = itertools.chain(self.unet.parameters(), self.text_encoder.parameters()) if self.args.train_text_encoder else self.unet.parameters()
                        self.accelerator.clip_grad_norm_(params_to_clip, self.args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    
                # 🌟 W&B 및 진행률 표시
                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

                if global_step >= self.args.max_train_steps:
                    break

        self.accelerator.wait_for_everyone()

    def save_model(self):
        """4단계: 훈련된 모델 안전하게 저장"""
        if self.accelerator.is_main_process:
            pipeline = DiffusionPipeline.from_pretrained(
                self.args.pretrained_model_name_or_path,
                unet=self.accelerator.unwrap_model(self.unet),
                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                revision=self.args.revision,
            )
            pipeline.save_pretrained(self.args.output_dir)
            logger.info(f"Model successfully saved to {self.args.output_dir}")
        self.accelerator.end_training()

    def run(self):
        """마스터 스위치: 모든 단계를 순서대로 실행!"""
        self.prepare_data_and_optimizers()
        self.train()
        self.save_model()


    # ========================================
    # 🛠️ Private Helper Methods
    # ========================================

    def _import_model_class(self):
        text_encoder_config = PretrainedConfig.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.args.revision,
        )
        if text_encoder_config.architectures[0] == "CLIPTextModel":
            from transformers import CLIPTextModel
            return CLIPTextModel
        else:
            raise ValueError("Only CLIPTextModel is supported in this refactored script.")

    def _generate_class_images(self):
        class_images_dir = Path(self.args.class_data_dir)
        class_images_dir.mkdir(parents=True, exist_ok=True)
        cur_class_images = len(list(class_images_dir.glob("*.jpg")) + list(class_images_dir.glob("*.png")))

        if cur_class_images >= self.args.num_class_images:
            return 

        num_new_images = self.args.num_class_images - cur_class_images
        torch_dtype = torch.float16 if self.accelerator.device.type == "cuda" else torch.float32
        pipeline = DiffusionPipeline.from_pretrained(
            self.args.pretrained_txt2img_model_name_or_path, torch_dtype=torch_dtype, safety_checker=None
        )
        pipeline.set_progress_bar_config(disable=True)
        pipeline.to(self.accelerator.device)

        sample_dataset = PromptDataset(self.args.class_prompt, num_new_images)
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=self.args.sample_batch_size)
        sample_dataloader = self.accelerator.prepare(sample_dataloader)

        for example in tqdm(sample_dataloader, desc="Generating class images", disable=not self.accelerator.is_local_main_process):
            images = pipeline(example["prompt"]).images
            for i, image in enumerate(images):
                hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                image.save(class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg")

        del pipeline
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _create_depth_images(self, paths):
        """독립되어 있던 함수를 클래스 안으로 우아하게 이사!"""
        pipeline = DiffusionPipeline.from_pretrained(
            self.args.pretrained_model_name_or_path,
            unet=self.accelerator.unwrap_model(self.unet),
            text_encoder=self.accelerator.unwrap_model(self.text_encoder),
            revision=self.args.revision,
        ).to(self.accelerator.device)
        
        for path in paths:
            all_files = list(Path(path).rglob("*.jpg")) + list(Path(path).rglob("*.png"))
            non_depth_files = [p for p in all_files if "_depth." not in p.name]
            
            for image_path in tqdm(non_depth_files, desc=f"Creating depths for {path}"):
                depth_path = get_depth_image_path(image_path)
                if depth_path.exists(): continue
                
                image_instance = Image.open(image_path).convert("RGB")
                image_tensor = pipeline.feature_extractor(image_instance, return_tensors="pt").pixel_values.to(self.accelerator.device)
                
                depth_map = pipeline.depth_estimator(image_tensor).predicted_depth
                depth_min, depth_max = torch.amin(depth_map, dim=[0,1,2], keepdim=True), torch.amax(depth_map, dim=[0,1,2], keepdim=True)
                depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
                
                depth_map_image = transforms.ToPILImage()(depth_map[0,:,:])
                depth_map_image.save(depth_path)
                
        vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
        del pipeline
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return vae_scale_factor

# 실행부
if __name__ == "__main__":
    # parser 등은 argparse나 omegaconf/hydra 로드 로직으로 채워주면 됨!
    # args = load_configs()
    
    # trainer = DreamBooth(args)
    # trainer.run()
    pass