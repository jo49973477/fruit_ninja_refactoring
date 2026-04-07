import os
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

import hydra
from omegaconf import DictConfig, OmegaConf

from diffusers import DDPMScheduler
from diffusers import StableDiffusionDepth2ImgPipeline
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

from utils.configs import FinetuneConfig



class DreamBoothFineTuning:
    def __init__(self, finetune_config: FinetuneConfig):
        self.cfg = finetune_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            print(f"🚀 SD model is assigned in {self.cfg.sd_model}...")
            self.pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
                self.cfg.sd_model
            ).to(self.device)
            self.pipe.set_progress_bar_config(disable=True)
            self.pipe.unet.enable_gradient_checkpointing()
        except Exception as e:
            raise RuntimeError(
                f"\n🚨 Failed to load {self.cfg.sd_model}!\n"
                f"🔥 The real cause: {e}"
            ) from e
            
        if self.cfg.load_dir is not None:
            self.pipe.load_lora_weights(self.cfg.load_dir)
            
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        self.pipe.unet = get_peft_model(self.pipe.unet, self.lora_config)
    
    
        
    def train_dreambooth(
        self,
        instance_images, # [B, C, H, W] 텐서
        instance_prompt: str, 
        desc_mention: str,
        class_images=None, 
        num_class_images=4 
    ):
        device = self.pipe.device
        
        # 🌟 0. [핵심 기능] class_images가 없으면 직접 생성합니다!!
        if class_images is None:
            print(f"🤖 삐리빅! class_images가 없습니다! '{self.cfg.class_prompt}' 프롬프트로 즉석 생성을 시작합니다!")
            generated_images = []
            
            # 🚨 주의: Depth2Img 모델은 기본적으로 입력 image가 필요해! 
            # 빈 도화지(회색 이미지 등)를 하나 만들어서 넣어주는 꼼수가 필요할 수 있어.
            # (만약 에러가 난다면, Text2Img 파이프라인으로 잠시 교체해서 뽑는 것도 방법이야!)
            dummy_image = torch.ones((1, 3, 512, 512), device=device) # 임시 도화지
            
            with torch.no_grad():
                for _ in range(num_class_images):
                    # 모델이 상상하는 "원래 오렌지" 생성
                    output = self.pipe(
                        prompt=self.cfg.class_prompt,
                        image=dummy_image, # Depth2Img용 더미
                        strength=1.0,      # 원본 무시하고 프롬프트대로만!
                        num_inference_steps=30
                    ).images[0]
                    generated_images.append(output)
            
            # 생성된 PIL 이미지들을 VAE가 좋아하는 텐서([-1, 1])로 변환!
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            class_images = torch.stack([transform(img) for img in generated_images]).to(device)

        # --- (이하 캡틴의 기존 코드와 동일) ---
        unet = self.pipe.unet
        text_encoder = self.pipe.text_encoder
        vae = self.pipe.vae
        noise_scheduler = self.pipe.scheduler
        
        optimizer = torch.optim.AdamW(unet.parameters(), lr=self.cfg.learning_rate)
        
        
        
        unet.train()

        # 2. 텍스트 임베딩 미리 뽑아두기 (속도 향상!)
        with torch.no_grad():
            # [Instance] "A vertical cross-section of a [V] orange"
            inst_inputs = self.pipe.tokenizer(instance_prompt, padding="max_length", truncation=True, return_tensors="pt").to(device)
            inst_embeds = text_encoder(inst_inputs.input_ids)[0]
            
            # [Class] "An orange"
            class_inputs = self.pipe.tokenizer(self.cfg.class_prompt, padding="max_length", truncation=True, return_tensors="pt").to(device)
            class_embeds = text_encoder(class_inputs.input_ids)[0]
        
        for epoch in tqdm(range(self.cfg.num_train_epochs), desc=desc_mention):
            perm = torch.randperm(instance_images.size(0))
            shuffled_instances = instance_images[perm]
            
            for i in range(0, shuffled_instances, self.cfg.train_batch_size):
            
                optimizer.zero_grad()
                
                batch_inst = shuffled_instances[i: i+self.cfg.train_batch_size]
                current_bsz = batch_inst.size(0)
                
                # 클래스 이미지(Prior)도 현재 배치 개수(current_bsz)에 맞춰서 랜덤하게 뽑아오기
                class_indices = torch.randint(0, class_images.size(0), (current_bsz,))
                batch_class = class_images[class_indices]
                
                # VAE로 이미지들을 잠재 공간(Latent)으로 압축
                with torch.no_grad():
                    latents_inst = vae.encode(batch_inst).latent_dist.sample() * vae.config.scaling_factor
                    latents_class = vae.encode(batch_class).latent_dist.sample() * vae.config.scaling_factor
                    
                noise_inst = torch.randn_like(latents_inst)
                noise_class = torch.randn_like(latents_class)
                
                # 텍스트 임베딩도 현재 배치 개수에 맞게 늘려주기
                inst_embeds_batch = inst_embeds.repeat(current_bsz, 1, 1)
                class_embeds_batch = class_embeds.repeat(current_bsz, 1, 1)
                
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (current_bsz,), device=device).long()
                class_timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (current_bsz,), device=device).long()

                noisy_latents_inst = noise_scheduler.add_noise(latents_inst, noise_inst, timesteps)
                noisy_latents_class = noise_scheduler.add_noise(latents_class, noise_class, class_timesteps)
                
                # Depth map 추출 (기존 코드와 동일하게 현재 배치 기준으로)
                depth_inst = self.pipe.prepare_depth_map(batch_inst, None, current_bsz, False, noisy_latents_inst.dtype, device)
                depth_class = self.pipe.prepare_depth_map(batch_class, None, current_bsz, False, noisy_latents_class.dtype, device)

                latent_model_input_inst = torch.cat([noisy_latents_inst, depth_inst], dim=1)
                latent_model_input_class = torch.cat([noisy_latents_class, depth_class], dim=1)
                
                # UNet 예측
                model_pred_inst = unet(latent_model_input_inst, timesteps, encoder_hidden_states=inst_embeds_batch).sample
                model_pred_class = unet(latent_model_input_class, class_timesteps, encoder_hidden_states=class_embeds_batch).sample
                
                # 4. ⚖️ [논문 핵심] Prior Preservation Loss 계산!
                loss_inst = F.mse_loss(model_pred_inst.float(), noise_inst.float(), reduction="mean")
                loss_class = F.mse_loss(model_pred_class.float(), noise_class.float(), reduction="mean")

                # 두 Loss 합치기
                loss = loss_inst + self.cfg.prior_loss_weight * loss_class

                # 역전파 및 4장 단위로 가중치 업데이트!
                loss.backward()
                optimizer.step()

        print("🤖 벼락치기 완료! 이제 SD가 오렌지 단면을 완벽하게 이해합니다!")
        
        # 학습된 UNet을 원래 파이프라인에 돌려줍니다.
        self.pipe.unet = unet
        
        torch.cuda.empty_cache()
        return self.pipe
    
    
    
    def save_weights(self):
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        self.pipe.unet.save_pretrained(self.cfg.save_dir)
        
        print(f"🤖 삐리빅! 저장 완료! 캡틴의 '오렌지 단면 비법 노트'가 [{self.cfg.save_dir}] 폴더에 안전하게 격리(?)되었습니다!")
    
    
    
    def load_images_from_directory(self, directory_path, image_size=512):
        """
        [로보코 특제] 폴더에서 이미지를 싹 긁어와서 SD 모델이 좋아하는 텐서로 변환합니다!
        """
        # 🌟 SD 모델 VAE가 좋아하는 입맛 레시피!
        # 1. 512x512로 크기 맞추기
        # 2. 파이토치 텐서로 변환 (0~1 사이 값으로 됨)
        # 3. 0~1 값을 -1~1 값으로 정규화! (이게 제일 중요!)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])

        image_tensors = []
        
        # 폴더 안의 파일들을 하나씩 꺼내봅니다.
        valid_extensions = (".png", ".jpg", ".jpeg")
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"🚨 '{directory_path}' directory does not exist!")

        for filename in sorted(os.listdir(directory_path)):
            if filename.lower().endswith(valid_extensions):
                image_path = os.path.join(directory_path, filename)
                try:
                    # 이미지를 열고 RGB로 변환 (알파 채널이 있으면 3채널로 깎아냄!)
                    image = Image.open(image_path).convert("RGB")
                    
                    # 텐서로 변환! [C, H, W]
                    tensor = transform(image)
                    image_tensors.append(tensor)
                except Exception as e:
                    print(f"⚠️ Cannot read '{filename}' file due to the dark power... {e}")

        if not image_tensors:
            raise ValueError(f"🚨 There is no usable images in the '{directory_path}' directory!")

        batch_tensor = torch.stack(image_tensors).to(self.device)
        
        
        return batch_tensor
    
    
    
    def finetune(self):
        horizontal_dir = os.path.join(self.cfg.image_dir, "horizontal")
        vertical_dir = os.path.join(self.cfg.image_dir, "vertical")
        
        horizontal_batch = self.load_images_from_directory(horizontal_dir)
        vertical_batch = self.load_images_from_directory(vertical_dir)
        
        if self.cfg.nickname is not None:
            vanilla_dir = os.path.join(self.cfg.image_dir, "vanilla")
            vanilla_batch = self.load_images_from_directory(vanilla_dir)
            
            horizontal_prompt = f"A horizontal cross-section of a {self.cfg.nickname} {self.cfg.class_prompt}"
            vertical_prompt = f"A vertical cross-section of a {self.cfg.nickname} {self.cfg.class_prompt}"
            
            print(f"🍎Let's start training of {self.cfg.nickname} in the class {self.cfg.class_prompt}!")
            
        else:
            horizontal_prompt = f"A horizontal cross-section of a {self.cfg.class_prompt}"
            vertical_prompt = f"A vertical cross-section of a {self.cfg.class_prompt}"
            
            print(f"🍎Let's start training of {self.cfg.class_prompt}!")
            
        if self.cfg.nickname is not None:
            self.train_dreambooth(
                                instance_images = vanilla_batch,
                                instance_prompt = self.cfg.class_prompt, 
                                desc_mention = f"🌟Teaching {self.cfg.nickname}, the new {self.cfg.class_prompt}...")
        
        self.train_dreambooth(
                            instance_images = vertical_batch,
                            instance_prompt = vertical_prompt, 
                            desc_mention = "🎥Horizontical view fine-tuning started...")
        
        self.train_dreambooth(
                            instance_images = horizontal_batch,
                            instance_prompt = horizontal_prompt, 
                            desc_mention = "🔪Vertical view fine-tuning started...")
        
        self.save_weights()



@hydra.main(version_base=None, config_path="config", config_name="finetune_config")
def main(cfg: FinetuneConfig):
    
    print("⚙️ [Hydra] 설정 파일 로드 완료!")
    
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    trainer_config = FinetuneConfig(**config_dict)
    trainer = DreamBoothFineTuning(trainer_config)
    trainer.finetune()



if __name__ == "__main__":
    main()
        