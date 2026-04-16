import argparse
import hashlib
import os
import gc # 가비지 컬렉터 추가
from pathlib import Path
import itertools
import json

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torchvision import transforms

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.utils.import_utils import is_xformers_available
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DiffusionPipeline
from transformers import AutoTokenizer, PretrainedConfig, DPTForDepthEstimation, DPTImageProcessor, CLIPTextModel

from utils.configs import FinetuneConfig

logger = get_logger(__name__)

# ==========================================================
# 🛠️ Helper Functions
# ==========================================================
def get_depth_image_path(normal_image_path):
    """원본 이미지 경로를 받아 Depth 이미지 경로를 반환합니다."""
    return normal_image_path.parent / f"{normal_image_path.stem}_depth.png"


def collate_fn(examples, with_prior_preservation=False):
    """데이터로더를 위한 배치 생성 함수"""
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

    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "depth_values": depth_values
    }



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


# ==========================================================
# 📦 Dataset Classes
# ==========================================================
class DreamBoothDataset(Dataset):
    def __init__(self, instance_data_root, tokenizer, vae_scale_factor=8, nickname="zxy", class_prompt="screw", size=512, class_data_root=None, prompt_json_dir=None):
        self.size = size
        self.tokenizer = tokenizer
        self.vae_scale_factor = vae_scale_factor
        self.nickname = nickname
        self.class_prompt = class_prompt
        
        # --- 1. Instance Data Load ---
        self.instance_data_root = Path(instance_data_root)
        all_instance_images = list(self.instance_data_root.rglob("*.jpg")) + list(self.instance_data_root.rglob("*.png"))
        self.instance_images_path = [p for p in all_instance_images if "_depth." not in str(p)]
        self.num_instance_images = len(self.instance_images_path)
        
        self.prompt_dict = {}
        if prompt_json_dir is not None:
            json_path = Path(prompt_json_dir)
            # 만약 디렉토리 경로만 줬다면 metadata.json을 알아서 찾도록!
            if json_path.is_dir():
                json_path = json_path / "metadata.json"
                
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.prompt_dict = json.load(f)
                print(f"🤖 삐리빅! JSON 파일 로드 완료! 총 {len(self.prompt_dict)}개의 정밀 프롬프트를 장착했습니다! ({json_path})")
            else:
                print(f"⚠️ [경고] prompt_json_dir이 주어졌지만 {json_path}를 찾을 수 없습니다. 기본 폴더명 프롬프트로 대체합니다.")

        # --- 2. Class Data Load (Prior Preservation) ---
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            all_class_images = list(self.class_data_root.rglob("*.jpg")) + list(self.class_data_root.rglob("*.png"))
            self.class_images_path = [p for p in all_class_images if "_depth." not in str(p)]
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None
            self._length = self.num_instance_images

        # --- 3. Transforms ---
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(size),
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
        
        # ==========================================
        # [A] Instance Image 처리 (JSON vs 폴더명 프롬프트)
        # ==========================================
        instance_idx = index % self.num_instance_images
        image_path = self.instance_images_path[instance_idx]
        depth_path = image_path.parent / f"{image_path.stem}_depth.png"
        
        instance_image = Image.open(image_path).convert("RGB")
        instance_depth_image = Image.open(depth_path).convert("L")
            
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_depth_images"] = self.depth_image_transforms(instance_depth_image)
        
        filename = image_path.name
        
        if self.prompt_dict and filename in self.prompt_dict:
            prompt = self.prompt_dict[filename]
        else:
            # 캡틴의 오리지널 폴더명 기반 동적 프롬프트 로직
            folder_name = image_path.parent.name
            if folder_name == "horizontal":
                prompt = f"A horizontal cross-section of a {self.nickname} {self.class_prompt}" if self.nickname is not None else f"A horizontal cross-section of a {self.class_prompt}"
            elif folder_name == "vertical":
                prompt = f"A vertical cross-section of a {self.nickname} {self.class_prompt}" if self.nickname is not None else f"A vertical cross-section of a {self.class_prompt}"
            else:
                prompt = f"A {self.nickname} {self.class_prompt}" if self.nickname is not None else f"A {self.class_prompt}"
            
        example["instance_prompt_ids"] = self.tokenizer(
            prompt, truncation=True, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
        ).input_ids

        # ==========================================
        # [B] Class Image 처리 (Prior Preservation)
        # ==========================================
        if self.class_data_root:
            class_idx = index % self.num_class_images
            class_image_path = self.class_images_path[class_idx]
            
            class_depth_path = class_image_path.parent / f"{class_image_path.stem}_depth.png"
            class_image = Image.open(class_image_path).convert("RGB")
            
            if not class_depth_path.exists():
                raise FileNotFoundError(
                    f"🚨 [ERROR] 클래스 이미지의 Depth 맵이 없습니다: {class_depth_path}\n"
                    "Depth2Img 모델을 훈련하려면 클래스 이미지(prior)도 반드시 뎁스 맵이 짝꿍으로 있어야 합니다!"
                )
                
            class_depth_image = Image.open(class_depth_path).convert("L")
                
            example["class_images"] = self.image_transforms(class_image)
            example["class_depth_images"] = self.depth_image_transforms(class_depth_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example



# ==========================================================
# 🚀 Main Trainer Class (Depth 자동 생성 로직 추가)
# ==========================================================
class DreamBoothTrainer:
    def __init__(self, args: FinetuneConfig):
        self.args = args
        self.setup_accelerator()
        self.generate_class_images_if_needed()
        self.generate_depth_images_if_needed() # 🌟 여기서 핵심 알고리즘 실행!
        self.setup_models()
        self.setup_optimizer()
        self.prepare_data()
        
        
    def setup_accelerator(self):
        if self.args.seed is not None:
            set_seed(self.args.seed)
            
        if self.args.output_dir is not None:
            Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
            
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)
        project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with="wandb", # or "wandb"
            project_config=project_config,
        )
        
        self.accelerator.init_trackers("dreambooth-training")
        
        if self.args.train_text_encoder and self.args.gradient_accumulation_steps > 1 and self.accelerator.num_processes > 1:
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )
    
      
        
    def generate_class_images_if_needed(self):
        """Prior Preservation을 위한 Class Image 자동 생성 로직"""
        
        if not self.args.with_prior_preservation:
            return

        class_images_dir = Path(self.args.class_data_dir)
        class_images_dir.mkdir(parents=True, exist_ok=True)

        # _depth가 안 붙은 순수 원본 이미지만 카운트
        all_images = list(class_images_dir.rglob("*.jpg")) + list(class_images_dir.rglob("*.png"))
        original_class_images = [p for p in all_images if "_depth." not in str(p)]
        cur_class_images = len(original_class_images)

        if cur_class_images >= self.args.num_class_images:
            logger.info(f"클래스 이미지가 충분합니다 ({cur_class_images}개). 생성을 건너뜁니다.")
            return

        num_new_images = self.args.num_class_images - cur_class_images
        logger.info(f"{num_new_images}개의 클래스 이미지 생성을 시작합니다...")

        torch_dtype = torch.float16 if self.accelerator.device.type == "cuda" else torch.float32
        
        # 🌟 주의: 여기서는 Depth 모델이 아니라 일반 텍스트-이미지 생성 모델을 씁니다.
        pipeline = DiffusionPipeline.from_pretrained(
            self.args.pretrained_txt2img_model_name_or_path, # e.g., "runwayml/stable-diffusion-v1-5"
            torch_dtype=torch_dtype,
            safety_checker=None,
            revision=self.args.revision,
        )
        pipeline.set_progress_bar_config(disable=True)
        pipeline.to(self.accelerator.device)

        # 이미지 생성 루프
        for i in tqdm(range(num_new_images), desc="Generating class images"):
            image = pipeline(f"A photo of {self.args.class_prompt}").images[0]
            hash_image = hashlib.sha1(image.tobytes()).hexdigest()
            image.save(class_images_dir / f"{cur_class_images + i}-{hash_image}.jpg")

        # 메모리 반환 (VRAM 확보)
        del pipeline
        torch.cuda.empty_cache()



    def generate_depth_images_if_needed(self):
        """🌟 원본 이미지로부터 Depth 맵을 추출하여 저장하는 캐싱 로직"""
        
        instance_data_root = Path(self.args.instance_data_dir)
        all_images = list(instance_data_root.rglob("*.jpg")) + list(instance_data_root.rglob("*.png"))
        original_images = [p for p in all_images if "_depth." not in str(p)]
        
        if self.args.with_prior_preservation and self.args.class_data_dir:
            class_data_root = Path(self.args.class_data_dir)
            all_images += list(class_data_root.rglob("*.jpg")) + list(class_data_root.rglob("*.png"))
            
        original_images = [p for p in all_images if "_depth." not in str(p)]
        
        # 새로 생성해야 할 Depth 맵이 있는지 확인
        images_needing_depth = [
            p for p in original_images 
            if not (p.parent / f"{p.stem}_depth.png").exists()
        ]
        
        if not images_needing_depth:
            logger.info("모든 Depth 이미지가 이미 존재합니다. 생성을 건너뜁니다.")
            return

        logger.info(f"{len(images_needing_depth)}개의 Depth 이미지 생성을 시작합니다...")
        
        # 1. SD Depth 모델에 내장된 DPT(MiDaS) 모듈 로드
        depth_estimator = DPTForDepthEstimation.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="depth_estimator"
        ).to(self.accelerator.device)
        
        feature_extractor = DPTImageProcessor.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="feature_extractor"
        )
        
        # 2. 추론 모드로 변환
        depth_estimator.eval()
        
        # 3. 이미지 순회하며 Depth 추출 및 저장
        for img_path in tqdm(images_needing_depth, desc="Generating Depth Maps"):
            image = Image.open(img_path).convert("RGB")
            
            # Feature extractor를 통해 텐서로 변환
            inputs = feature_extractor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.accelerator.device)
            
            with torch.no_grad(), torch.autocast(self.accelerator.device.type):
                outputs = depth_estimator(pixel_values)
                predicted_depth = outputs.predicted_depth
                
            # 깊이 맵 크기를 원본 이미지 크기로 보간(Interpolation)
            predicted_depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(image.height, image.width),
                mode="bicubic",
                align_corners=False,
            )
            
            # 시각화 가능한 0~255 이미지 형태로 정규화(Normalize)
            depth_min = torch.amin(predicted_depth, dim=[1, 2, 3], keepdim=True)
            depth_max = torch.amax(predicted_depth, dim=[1, 2, 3], keepdim=True)
            depth_normalized = (predicted_depth - depth_min) / (depth_max - depth_min)
            depth_normalized = (depth_normalized * 255.0).squeeze().cpu().numpy().astype(np.uint8)
            
            # 저장
            depth_image = Image.fromarray(depth_normalized)
            save_path = img_path.parent / f"{img_path.stem}_depth.png"
            depth_image.save(save_path)
            
        # 4. 메모리 정리 (VRAM 확보를 위해 아주 중요함!)
        logger.info("Depth 이미지 생성 완료! VRAM 확보를 위해 Depth 모델을 메모리에서 해제합니다.")
        del depth_estimator
        del feature_extractor
        torch.cuda.empty_cache()
        gc.collect()



    def setup_models(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.pretrained_model_name_or_path, 
            subfolder="tokenizer", 
            revision=self.args.revision
        )
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.args.pretrained_model_name_or_path, 
            subfolder="text_encoder", 
            revision=self.args.revision
        )
        
        self.vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae", revision=self.args.revision)
        self.unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet", revision=self.args.revision)
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="scheduler")

        self.vae.requires_grad_(False)
        if not self.args.train_text_encoder:
            self.text_encoder.requires_grad_(False)

        if self.args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.args.train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()
                
        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()



    def setup_optimizer(self):
        if self.args.scale_lr:
            self.args.learning_rate = (
                self.args.learning_rate * self.args.gradient_accumulation_steps * self.args.train_batch_size * self.accelerator.num_processes
            )
        
        if self.args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_cls = bnb.optim.AdamW8bit
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )
            
        else:
            optimizer_cls = torch.optim.AdamW
        
        params_to_optimize = (
            itertools.chain(self.unet.parameters(), self.text_encoder.parameters()) 
            if self.args.train_text_encoder else self.unet.parameters()
        )
        self.optimizer = optimizer_cls(
            params_to_optimize, lr=self.args.learning_rate, 
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay, eps=self.args.adam_epsilon,
        )



    def _create_depth_images(self):
        """기존 코드의 create_depth_images 로직을 클래스 내부로 통합"""
        # Pipeline을 로드하여 Depth 맵을 생성하고 디스크에 저장하는 로직
        pass # (원래 코드의 pipeline.depth_estimator 로직 복붙)
        return 8 # vae_scale_factor (일반적으로 8)



    def prepare_data(self):
        from diffusers.optimization import get_scheduler # 상단 import에 추가해도 됨
        
        # 1. 아까 만든 쓰레기 함수(_create_depth_images) 호출 삭제! vae_scale_factor는 8 고정!
        self.train_dataset = DreamBoothDataset(
            instance_data_root=self.args.instance_data_dir,
            tokenizer=self.tokenizer,
            vae_scale_factor=8, # 🌟 8로 고정
            size=self.args.resolution,
            nickname=self.args.nickname, 
            class_prompt=self.args.class_prompt,
            class_data_root= self.args.class_data_dir if self.args.with_prior_preservation else None,
            prompt_json_dir = self.args.prompt_json_dir
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.args.train_batch_size, shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, self.args.with_prior_preservation),
            num_workers=1,
        )

        # 2. 🌟 잃어버린 스케줄러(lr_scheduler) 생성 로직 복구!
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.args.max_train_steps * self.args.gradient_accumulation_steps,
        )

        # 3. 🌟 [가장 중요] Accelerator에게 모든 걸 맡기기 (VRAM 최적화 및 분산 처리의 핵심)
        if self.args.train_text_encoder:
            self.unet, self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
                self.unet, self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler
            )
        else:
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
                self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
            )

    def train(self):
        # 학습 가중치 타입 설정
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            
        self.vae.to(self.accelerator.device, dtype=weight_dtype)
        
        # 텍스트 인코더도 학습 안 할 거면 같이 멱살 잡고 끌어올려야 해!
        if not self.args.train_text_encoder:
            self.text_encoder.to(self.accelerator.device, dtype=weight_dtype)

        # 진행바 설정
        global_step = 0
        first_epoch = 0
        progress_bar = tqdm(
            range(global_step, self.args.max_train_steps), 
            disable=not self.accelerator.is_local_main_process
        )
        progress_bar.set_description("Steps")

        # 🌟 본격적인 훈련 루프 시작
        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.unet.train()
            if self.args.train_text_encoder:
                self.text_encoder.train()
                
            for step, batch in enumerate(self.train_dataloader):
                # Checkpoint Resume 로직 (필요시)
                if self.args.resume_from_checkpoint and epoch == first_epoch and step < self.args.resume_step:
                    if step % self.args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                with self.accelerator.accumulate(self.unet):
                    # 1. Encoding latent vector
                    latents = self.vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215

                    # 2. Noice scheduling, inserting noise latent
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                    
                    # 3. combining noise latents and depth values
                    noisy_latents = torch.cat([noisy_latents, batch["depth_values"].to(dtype=weight_dtype)], dim=1)

                    # 4. extracting hidden states
                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

                    # 5. predicting noise
                    model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # 6.setting prediction type
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

                    # 7. calculating loss
                    if self.args.with_prior_preservation:
                        # 배치 절반은 Instance, 절반은 Class 이므로 잘라서(Chunk) 따로 계산
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                        # 최종 Loss = 원본 Loss + (가중치 * 클래스 Loss)
                        loss = loss + self.args.prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # 8.backpropagation using accelerator
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
                            if self.args.train_text_encoder else self.unet.parameters()
                        )
                        self.accelerator.clip_grad_norm_(params_to_clip, self.args.max_grad_norm)
                        
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # 9. saving logging and checkpoint
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % self.args.checkpointing_steps == 0:
                        if self.accelerator.is_main_process:
                            # (1) 기존 방식: 나중에 학습 재개(Resume)를 위한 원시 부품 저장
                            save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
                            self.accelerator.save_state(save_path)
                            
                            # ==========================================================
                            # 🌟 (2) 추가된 방식: 즉시 추론 가능한 '완성품 파이프라인' 저장!
                            # ==========================================================
                            from diffusers import StableDiffusionDepth2ImgPipeline
                            
                            logger.info(f"중간 평가용 파이프라인 조립 중... (Step: {global_step})")
                            intermediate_pipeline = StableDiffusionDepth2ImgPipeline.from_pretrained(
                                self.args.pretrained_model_name_or_path,
                                unet=self.accelerator.unwrap_model(self.unet),
                                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                                revision=self.args.revision,
                                torch_dtype=weight_dtype,
                            )
                            
                            # checkpoint-1000 폴더 안에 'pipeline' 이라는 하위 폴더를 만들어서 예쁘게 포장!
                            pipeline_save_path = os.path.join(save_path, "pipeline")
                            intermediate_pipeline.save_pretrained(pipeline_save_path)
                            
                            logger.info(f"✅ Saved state to {save_path} and full pipeline to {pipeline_save_path}")
                            
                            # VRAM 터짐 방지!
                            del intermediate_pipeline
                            torch.cuda.empty_cache()
                            # ==========================================================

                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

                if global_step >= self.args.max_train_steps:
                    break
            
               
            if epoch % 10 == 0 and self.accelerator.is_main_process:
                logger.info(f"Epoch {epoch}: Running validation and logging to WandB...")
                
                from diffusers import StableDiffusionDepth2ImgPipeline
                
                # 1. 평가 모드로 전환
                self.unet.eval()
                if self.args.train_text_encoder:
                    self.text_encoder.eval()

                # 2. 검증용 파이프라인 임시 생성 (VRAM을 위해 unwrap 필수!)
                val_pipeline = StableDiffusionDepth2ImgPipeline.from_pretrained(
                    self.args.pretrained_model_name_or_path,
                    unet=self.accelerator.unwrap_model(self.unet),
                    text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                    revision=self.args.revision,
                    torch_dtype=weight_dtype,
                )
                val_pipeline = val_pipeline.to(self.accelerator.device)
                val_pipeline.set_progress_bar_config(disable=True)

                # 3. 검증에 사용할 기준 이미지 로드 (캡틴의 데이터셋 첫 번째 이미지 사용!)
                val_image_path = self.train_dataset.instance_images_path[0]
                val_image = Image.open(val_image_path).convert("RGB")
                
                # 테스트 프롬프트 (데이터셋에 설정된 nickname과 class 사용)
                val_prompt = f"A {self.train_dataset.nickname} {self.train_dataset.class_prompt}" if self.train_dataset.nickname is not None else f"A {self.train_dataset.class_prompt}"

                # 4. 이미지 생성!
                with torch.autocast(self.accelerator.device.type):
                    val_result_image = val_pipeline(
                        prompt=val_prompt,
                        image=val_image,
                        num_inference_steps=30,
                        strength = 0.8
                    ).images[0]

                # 5. Accelerator를 통해 WandB에 이미지 업로드
                for tracker in self.accelerator.trackers:
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "validation/image": wandb.Image(val_result_image, caption=f"Epoch {epoch} | {val_prompt}"),
                                "validation/original_reference": wandb.Image(val_image, caption="Original Reference")
                            },
                            step=global_step
                        )

                # 6. 🚨 [매우 중요] VRAM 폭발 방지를 위한 메모리 초기화!
                del val_pipeline
                torch.cuda.empty_cache()
                
                # 다시 훈련 모드로 복귀
                self.unet.train()
                if self.args.train_text_encoder:
                    self.text_encoder.train()
            # ====================================================================
        
        
            if global_step >= self.args.max_train_steps:
                break

        self.accelerator.wait_for_everyone()

        # 10. 학습 종료 후 최종 Pipeline 저장
        if self.accelerator.is_main_process:
            from diffusers import StableDiffusionDepth2ImgPipeline # 🌟 반드시 Depth 파이프라인으로 로드!
            
            pipeline = StableDiffusionDepth2ImgPipeline.from_pretrained(
                self.args.pretrained_model_name_or_path,
                unet=self.accelerator.unwrap_model(self.unet),
                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                revision=self.args.revision,
            )
            pipeline.save_pretrained(self.args.output_dir)

        self.accelerator.end_training()


@hydra.main(version_base=None, config_path="config", config_name="finetune")
def hydra_main(cfg: FinetuneConfig):
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    config = FinetuneConfig(**config_dict)
    trainer = DreamBoothTrainer(config)
    trainer.train()

    

if __name__ == "__main__":
    hydra_main()