import os
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from plyfile import PlyData
import numpy as np
from tqdm import tqdm
from pytorch_msssim import ssim

import hydra
from omegaconf import DictConfig, OmegaConf

import gsplat
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from diffusers import StableDiffusionDepth2ImgPipeline

from utils.configs import TrainerConfig, PhysicsConfig
from utils.transformation import *
import torchvision


class Trainer:
    
    def __init__(self, trainer_config: TrainerConfig):
        self.devices = self._get_optimal_devices()
        self.device = self.devices[0]
        
        self.trainer_cfg = trainer_config
        self.background = torch.tensor([1, 1, 1], dtype=torch.float32).to(self.device)
        
        self.gaussians_save = self.load_ply_to_gsplat(self.trainer_cfg.gaussian_path, self.device)
        self.gaussians_orig = self.load_ply_to_gsplat(self.trainer_cfg.gaussian_orig, self.device)
        
        for k in self.gaussians_save:
            # 🌟 원본 텐서 자체에 기울기 추적 기능을 켭니다. (가장 중요!!)
            self.gaussians_save[k].requires_grad_(True)
        
        self.transform = T.ToTensor()
        
        dev_vertical, dev_horizontal = self.devices[-2], self.devices[-1]
        
        try:
            print(f"🚀 Vertical SD model is assigned in {dev_vertical}...")
            self.pipe_vertical = StableDiffusionDepth2ImgPipeline.from_pretrained(
                self.trainer_cfg.sd_model_vertical
            ).to(dev_vertical)
            self.pipe_vertical.set_progress_bar_config(disable=True)
        except Exception as e:
            
            raise RuntimeError(
                f"\n🚨 Failed to load {self.trainer_cfg.sd_model_vertical}!\n"
                f"🔥 The real cause: {e}"
            ) from e

        try:
            print(f"🚀 Horizontical SD model is assigned in {dev_horizontal}...")
            self.pipe_horizontal = StableDiffusionDepth2ImgPipeline.from_pretrained(
                self.trainer_cfg.sd_model_horizontal
            ).to(dev_horizontal)
            self.pipe_horizontal.set_progress_bar_config(disable=True)
        except Exception as e:
            
            raise RuntimeError(
                f"\n🚨 Failed to load {self.trainer_cfg.sd_model_horizontal}!\n"
                f"🔥 The real cause: {e}"
            ) from e
            
        
        self.strategy = DefaultStrategy(
            refine_start_iter=100,
            refine_stop_iter=1500,
            verbose=True
        )
        
        self.strategy_state = self.strategy.initialize_state()
        
        self.optimizer = {k: optim.Adam([v], lr=self.trainer_cfg.lrs[k], eps=1e-15) for k, v in self.gaussians_save.items()}
        
        self.global_step = 0
    
    
    
    def load_ply_to_gsplat(self, ply_path: str, device = "cuda") -> dict[str, nn.Parameter]:
        
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"🚨The file {ply_path} is not found!🚨\n")
        
        plydata = PlyData.read(ply_path)
        v = plydata['vertex']

        means = np.stack((v['x'], v['y'], v['z']), axis=-1)
        scales = np.stack((v['scale_0'], v['scale_1'], v['scale_2']), axis=-1)
        quats = np.stack((v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']), axis=-1)
        opacities = v['opacity'][..., np.newaxis]
        colors = np.stack((v['f_dc_0'], v['f_dc_1'], v['f_dc_2']), axis=-1)

        return {
            "means": torch.tensor(means, dtype=torch.float32, device= device),
            "scales": torch.tensor(scales, dtype=torch.float32, device=device),
            "quats": torch.tensor(quats, dtype=torch.float32, device=device),
            "opacities": torch.tensor(opacities, dtype=torch.float32, device=device),
            "colors": torch.tensor(colors, dtype=torch.float32, device=device)
        }



    def _get_optimal_devices(self, num_models=2):
        num_gpus = torch.cuda.device_count()
        
        if num_gpus == 0:
            print("⚠️ You have no GPU and it will be run with only CPU. Cheer up bro")
            return ["cpu"] * (num_models + 1)
        
        if num_gpus == 1:
            print("⚠️ You only have 1 GPU. All models will be squeezed into cuda:0")
            return ["cuda:0"] * (num_models + 1)
            
        main_device = ["cuda:0"]
        sub_gpus = list(range(1, num_gpus)) 
        allocated_devices = [f"cuda:{sub_gpus[i % len(sub_gpus)]}" for i in range(num_models)]
        return main_device + allocated_devices
    
    
    
    def create_3d_grid(self, gaussians_dict: dict, grid_size: tuple):
        """Makes 3D grid from the gaussian dictionary."""
        xyz = gaussians_dict["means"]  # Shape (N, 3)
        
        # Get the min and max coordinates to define the grid boundaries
        min_coords = xyz.min(dim=0)[0]
        max_coords = xyz.max(dim=0)[0]
        
        # Calculate the dimensions of each grid cell
        cell_dimensions = (max_coords - min_coords) / torch.tensor(grid_size, device=self.device)

        # Create a dictionary to hold the grid
        grid = {}

        # Iterate over each gaussian and assign it to a grid cell
        for idx in range(xyz.size(0)):
            # Determine the grid cell for the current point
            cell_coords = ((xyz[idx] - min_coords) / cell_dimensions).floor().long()
            cell_key = tuple(cell_coords.tolist())
            
            if cell_key not in grid:
                grid[cell_key] = []
            
            grid[cell_key].append(idx)

        return grid



    def smooth_gaussians_in_grid(self, gaussians_dict: dict, grid: dict):
        """Based on the grid information, smoothing the features like colours."""
        
        scales = gaussians_dict["scales"] 
        rotations = gaussians_dict["quats"]  
        features = gaussians_dict["colors"]  # 원본의 _features_dc 대신 colors 사용!

        # Create tensors to hold the smoothed values
        smoothed_features = torch.zeros_like(features)

        counts = torch.zeros(len(scales), dtype=torch.int)

        for cell_key, indices in grid.items():
            if len(indices) > 0:
                cell_features = features[indices]
                avg_features = torch.mean(cell_features, dim=0).squeeze(0)
                smoothed_features[indices] = avg_features
                counts[indices] += 1
        
        with torch.no_grad():
            gaussians_dict["colors"].copy_(smoothed_features)
    
    
    
    def manage_densification_and_smoothing(self, epoch: int):
        """Gaussian Densification and Smoothing is managed here."""

        # --- 2. Grid Smoothing (101 에폭마다) ---
        if epoch > 0 and epoch % 101 == 0:
            print(f"\n🌟 [Epoch {epoch}] 3D Grid Smoothing!")
            
            with torch.no_grad():
                # 아까 캡틴의 Trainer 클래스에 예쁘게 이식한 그 엘리트 함수들을 호출!
                # (주의: 전역 변수 gaussians 대신 self.gaussians_save 딕셔너리 사용!)
                grid = self.create_3d_grid(self.gaussians_save, grid_size=(512, 512, 512))
                self.smooth_gaussians_in_grid(self.gaussians_save, grid)
    
    
    
    def _get_gsplat_camera_matrices(self, azimuth: float, elevation: float = 0.0):
        """Takes azimuth and elevation as input and produces w2c and K sight"""
        
        # 1. 카메라 해상도 및 반경 (config에서 가져온다고 가정!)
        width = self.trainer_cfg.image_size
        height = self.trainer_cfg.image_size
        radius = self.trainer_cfg.init_radius
        
        # 2. 각도를 라디안으로 변환
        azimuth_rad = torch.deg2rad(torch.tensor(float(azimuth)))
        elevation_rad = torch.deg2rad(torch.tensor(float(elevation)))
        
        # 3. 구면 좌표계 -> 3D 데카르트 좌표계 (카메라 위치 x, y, z)
        # 수학 시간! (cos, sin 섞어서 원을 그리며 도는 위치 잡기)
        cam_x = radius * torch.cos(elevation_rad) * torch.sin(azimuth_rad)
        cam_y = radius * torch.sin(elevation_rad)
        cam_z = radius * torch.cos(elevation_rad) * torch.cos(azimuth_rad)
        cam_pos = torch.tensor([cam_x, cam_y, cam_z], device=self.device)
        
        # 4. View Matrix (4x4) 만들기 - "카메라가 물체(0,0,0)를 바라보게 세팅"
        # (간단한 LookAt 수학 연산)
        forward = -cam_pos / torch.norm(cam_pos) # 원점을 바라보는 방향
        up = torch.tensor([0.0, 1.0, 0.0], device=self.device) # 위쪽 방향
        right = torch.linalg.cross(up, forward)
        right = right / torch.norm(right)
        up = torch.linalg.cross(forward, right)
        
        viewmat = torch.eye(4, device=self.device)
        viewmat[:3, 0] = right
        viewmat[:3, 1] = up
        viewmat[:3, 2] = forward
        viewmat[:3, 3] = cam_pos
        
        # 5. K Matrix (3x3 인트린직) - "시야각(FOV) 세팅"
        fov_y = torch.deg2rad(torch.tensor(60.0)) # 기본 FOV 60도
        focal = (height / 2.0) / torch.tan(fov_y / 2.0)
        
        K = torch.tensor([
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0]
        ], device=self.device)
        
        # gsplat은 카메라 이동을 위해 viewmat의 역행렬(World-to-Camera)을 주로 씁니다.
        # 방향에 따라 viewmat.inverse()를 써야 할 수도 있습니다!
        world_to_cam = torch.linalg.inv(viewmat) 
        
        return world_to_cam, K
    
    
    def _convert_SH_to_RGB(self, shs: torch.Tensor,):
        """SH degree 0 텐서를 [0~1] 사이의 진짜 RGB 색상으로 변환합니다!"""
        # SH0의 마법의 상수 (수학적으로 정해진 값!)
        SH_C0 = 0.28209479177387814
        
        # 색상 계산: RGB = SH * C0 + 0.5
        colors = shs * SH_C0 + 0.5
        
        # RGB 값이 0 이하로 가거나 1을 넘지 않게 꽉 잡아주기(Clamp)
        return torch.clamp(colors, min=0.0, max=1.0)

    
    
    
    def _generate_sds_reference(self, render_image: torch.Tensor, step: int, mode: str) -> Image.Image:
        # 1. 설정 준비
        pipe_prompt_nega = {
            "vertical": (self.pipe_vertical, self.trainer_cfg.vertical_prompt, self.trainer_cfg.vertical_negative_prompt),
            "horizontal": (self.pipe_horizontal, self.trainer_cfg.horizontal_prompt, self.trainer_cfg.horizontal_negative_prompt)
        }
        pipe, prompt, negative_prompt = pipe_prompt_nega[mode]
        device = pipe.device

        # 2. 텍스트 임베딩 미리 추출 (매 스텝마다 하면 느리니까!)
        text_inputs = pipe.tokenizer(
            [prompt, negative_prompt], # 🌟 리스트로 2개를 묶어주세요!
            padding="max_length", 
            max_length=pipe.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        text_embeddings = pipe.text_encoder(text_inputs.input_ids)[0]
        
        # 3. 이미지를 잠재 공간(Latent)으로 변환
        image_tensor = pipe.image_processor.preprocess(render_image.detach().permute(2, 0, 1).unsqueeze(0)).to(device)
        with torch.no_grad():
            init_latents = pipe.vae.encode(image_tensor).latent_dist.sample()
            init_latents = init_latents * pipe.vae.config.scaling_factor

        # 4. 최적화 준비
        init_latents = init_latents.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([init_latents], lr=0.03) # 속살 생성을 위해 적절한 LR
        
        min_t = int(pipe.scheduler.config.num_train_timesteps * 0.02) # 보통 20
        max_t = int(pipe.scheduler.config.num_train_timesteps * 0.98) # 보통 980
        
        # 🌟 [수동 SDS 루프 시작]
        for _ in range(self.trainer_cfg.sds_steps): # 5번 정도만 깎아봅시다!
            optimizer.zero_grad()
            
            
            t = torch.randint(min_t, max_t, (1,), device=device).long()
            
            # (2) 노이즈 생성 및 주입
            noise = torch.randn_like(init_latents)
            latents_noisy = pipe.scheduler.add_noise(init_latents, noise, t)
            
            depth_mask = pipe.prepare_depth_map(
                image=image_tensor,
                depth_map=None, # 직접 입력 대신 모델이 예측하게 함
                batch_size=1,
                do_classifier_free_guidance=True, # CFG 미사용 시 False
                dtype=init_latents.dtype,
                device=device,
            )
            
            # 2️⃣ [추가] 4채널 Latent와 1채널 Depth를 합체! (dim=1 방향으로)
            # 결과: [1, 5, 64, 64] 형태가 됩니다.
            latent_model_input = torch.cat([latents_noisy] * 2)
            latent_model_input = torch.cat([latent_model_input, depth_mask], dim=1)
            
            # (3) UNet으로 노이즈 예측
            with torch.no_grad():
                noise_pred_all = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            noise_pred_text, noise_pred_uncond = noise_pred_all.chunk(2)
            noise_pred = noise_pred_uncond + self.trainer_cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)
            grad = noise_pred - noise
            
            # (5) Latent 업데이트 (직접 조각하기)
            target = (init_latents - grad).detach()
            loss = 0.5 * F.mse_loss(init_latents.float(), target, reduction='sum')
            loss.backward()
            optimizer.step()

        # 5. 최종 결과물 복구
        with torch.no_grad():
            init_latents = init_latents / pipe.vae.config.scaling_factor
            output_image_tensor = pipe.vae.decode(init_latents).sample
            output_image = pipe.image_processor.postprocess(output_image_tensor)[0]
            
        return output_image
    
    
    
    def _get_vertical_slice_mask(self, pos: torch.Tensor, viewmat: torch.Tensor):
        """
        [수직 뷰 마스크 - 일명 '세로 반갈죽 컷' 🍎]
        오렌지의 정중앙(0, 0, 0)을 지나며 카메라를 마주 보는 평면을 생성해,
        카메라 앞쪽 절반을 싹둑 잘라내서 '세로 단면'을 노출시킵니다!
        """
        
        c2w = torch.linalg.inv(viewmat)
        forward_dir = c2w[:3, 2].to(self.device)
        center_pos = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        vec_to_pos = pos - center_pos
        depth_diff = torch.matmul(vec_to_pos, forward_dir)
        mask = depth_diff >= -0.01
        
        return mask
    
    

    def _get_horizontal_slice_mask(self, pos: torch.Tensor, viewmat: torch.Tensor, center_pos: torch.Tensor, slice_thickness: float):
        """
        [수평 단면 마스크 - 일명 '프루트 닌자 컷' 🔪]
        카메라가 내려다보는 방향을 칼날 평면의 법선(Normal) 벡터로 삼아,
        칼날(center_pos)보다 카메라 쪽에 가까운 껍질들을 싹둑 날려버립니다!
        """
        # 1. World-to-Camera 행렬(viewmat)을 뒤집어서 Camera-to-World (C2W) 획득!
        c2w = torch.linalg.inv(viewmat)
        
        # 2. 카메라가 바라보는 '정면(Forward) 방향' 벡터 추출
        # (캡틴이 만든 카메라 수학 공식에 따르면, C2W 행렬의 3번째 열(인덱스 2)이 Forward 벡터야!)
        forward_dir = c2w[:3, 2].to(self.device)
        
        # 3. 기준점(칼날의 중심)에서 각 가우시안 점(pos)으로 향하는 방향 벡터 계산
        # center_pos가 CPU 텐서나 리스트일 수 있으므로 확실하게 device로 올려줌!
        if not isinstance(center_pos, torch.Tensor):
            center_pos = torch.tensor(center_pos, device=self.device)
        center_pos = center_pos.to(self.device)
        
        vec_to_pos = pos - center_pos
        
        # 4. 내적(Dot Product)의 마법! 
        # 점들이 칼날 평면을 기준으로 카메라에서 먼 쪽(속살)인지, 가까운 쪽(껍질)인지 판별해.
        # 결과가 양수(+)면 카메라가 바라보는 방향 쪽(남겨야 할 속살), 음수(-)면 카메라 뒤쪽(잘라낼 껍질)!
        depth_diff = torch.matmul(vec_to_pos, forward_dir)
        
        # 5. 잘라내기! 
        # 수학적으로는 depth_diff >= 0 이면 되지만, 절단면이 너무 칼같으면 렌더링 시 구멍이 날 수 있어.
        # 그래서 slice_thickness 절반만큼의 여유 버퍼(buffer)를 줘서 살짝 도톰하게 남기는 게 꿀팁이야!
        mask = depth_diff >= -(slice_thickness / 2.0)
        
        return mask
    
    
    
    def get_loss(self, img1: torch.Tensor, img2: torch.Tensor, ssim_weight: float = 0.7) -> torch.Tensor:
        """
        [로보코 특제 통합 Loss 함수]
        렌더링 이미지와 정답 이미지의 차원(Shape)을 알아서 찰떡같이 맞춘 뒤,
        SSIM과 MSE를 조합하여 최종 Loss를 반환합니다! 차원 요괴 완벽 차단! 🛡️
        """
        # 1. img1 (gsplat 렌더링): 보통 [H, W, 3] -> 파이토치 표준 [1, 3, H, W] 로 변환
        if img1.dim() == 3 and img1.shape[-1] == 3:
            img1 = img1.permute(2, 0, 1).unsqueeze(0)
        elif img1.dim() == 3: # 이미 [3, H, W]라면
            img1 = img1.unsqueeze(0)
            
        # 2. img2 (SD 레퍼런스): 보통 [3, H, W] -> 파이토치 표준 [1, 3, H, W] 로 변환
        if img2.dim() == 3 and img2.shape[0] == 3:
            img2 = img2.unsqueeze(0)
        elif img2.dim() == 3 and img2.shape[-1] == 3: # 혹시 [H, W, 3] 이라면
            img2 = img2.permute(2, 0, 1).unsqueeze(0)

        # 3. 픽셀 값 범위 안전띠 꽉 매기! (0.0 ~ 1.0)
        img1 = img1.clamp(0.0, 1.0)
        img2 = img2.clamp(0.0, 1.0)

        # 4. SSIM Loss 계산 (1에 가까울수록 좋으니 1에서 뺌!)
        ssim_loss = 1.0 - ssim(img1, img2, data_range=1.0, size_average=True)
        
        # 5. MSE Loss 계산 (이제 차원이 똑같아서 절대 에러 안 남!)
        mse_loss = F.mse_loss(img1, img2)

        # 6. 가중치(기본값 0.7 대 0.3)로 예쁘게 섞어서 반환!
        total_loss = (ssim_weight * ssim_loss) + ((1.0 - ssim_weight) * mse_loss)
        
        return total_loss


    
    def _train_vertical_views(self, epoch: int):
        
        pbar = tqdm(range(30))
        
        for i in pbar:
            
            viewmat, K = self._get_gsplat_camera_matrices(i * 12, elevation=0) 
            
            mask = self._get_vertical_slice_mask(self.gaussians_save["means"], viewmat)
            
            means_cs = self.gaussians_save["means"][mask]
            quats_cs = self.gaussians_save["quats"][mask]
            scales_cs = self.gaussians_save["scales"][mask]
            opacities_cs = self.gaussians_save["opacities"][mask]
            shs_cs = self.gaussians_save["colors"][mask]
            
            means_cs.requires_grad_(True)
            
            colors_precomp_cs = self._convert_SH_to_RGB(shs_cs)
            scales_activated = torch.exp(scales_cs) 
            quats_activated = F.normalize(quats_cs, p=2, dim=-1) 
            opacity_activated = torch.sigmoid(opacities_cs).squeeze(-1)
            
            render_image, render_alpha, render_info = rasterization(
                means= means_cs,
                quats= quats_activated,                
                scales= scales_activated,             
                opacities= opacity_activated,
                colors= colors_precomp_cs,
                viewmats= viewmat.unsqueeze(0), 
                Ks= K.unsqueeze(0),             
                width= self.trainer_cfg.image_size,
                height= self.trainer_cfg.image_size,
                packed = True,
                backgrounds= self.background
            )
            # =========================================================
            
            # 4. SD 모델 뎁스 추출
            render_image = render_image.squeeze(0)
            depth_map = F.interpolate(render_image.permute(2, 0, 1).unsqueeze(0), size=(384, 384), mode='bilinear')[0]
            depth_map = depth_map.unsqueeze(0).to(self.devices[-1])
            
            self._save_render_image(render_image, f"v{i}_render")
            
            with torch.no_grad():
                predicted_depth = self.pipe_vertical.depth_estimator(depth_map).predicted_depth
                depth_resized = F.interpolate(predicted_depth.unsqueeze(0), size=(512, 512), mode='bilinear').squeeze(0)

            # 5. 레퍼런스 이미지 생성 및 타겟 세팅
            if epoch % self.trainer_cfg.sds_per_epoch == 0:
                ref_img = self._generate_sds_reference(render_image, step= self.trainer_cfg.sds_per_epoch - epoch//100, mode="vertical")
                self._save_ref_image(ref_img, f"v{i}_ref")
            else:
                ref_img = self._load_cached_reference(f"v{i}_ref.png")
            
            ground_truth = self.transform(ref_img).to(self.device)
            loss = self.get_loss(render_image, ground_truth)
            
            
            # 6. gsplat 비서 보고 및 역전파!
            self.strategy.step_pre_backward(
                params=self.gaussians_save,
                optimizers=self.optimizer,
                state=self.strategy_state,
                step=self.global_step, 
                info=render_info,  
            )
            
            render_info["means2d"].retain_grad()
            loss.backward()
            
            self.strategy.step_post_backward(
                params=self.gaussians_save,
                optimizers= self.optimizer,
                state=self.strategy_state,
                step=self.global_step,
                info=render_info,
                packed= True
            )
            
            for opt in self.optimizer.values():
                opt.step()
                opt.zero_grad()
            
            self.global_step += 1 
            
            pbar.set_description(f"🎥 [수직 뷰] {i}/30 앵글 촬영 및 학습 시작!: LOSS: {loss.item():.5f}")
            
            torch.cuda.empty_cache() 
    
    
    def _get_horizontal_slice_centers(self, steps: int = 70):
        """
        [로보코 특제 슬라이서] 🔪🍊
        가우시안 점들의 전체 높이(Z축) 범위를 계산해서,
        속살을 파먹을 슬라이스 중심점들과 1회분 두께를 반환합니다!
        """
        # 1. 전체 가우시안 점들의 3D 위치 가져오기
        pos = self.gaussians_save["means"]
        
        z_coords = pos[:, 2] 
        
        z_min = z_coords.min().item()
        z_max = z_coords.max().item()
        slice_thickness = (z_max - z_min) / steps
        slice_centers = torch.linspace(z_min, z_max, steps).to(self.device)
        
        return slice_centers, slice_thickness
            
            
    def _train_horizontal_views(self, epoch: int):
        """Phase 2: 카메라를 밀어 넣으며 가우시안의 속살(수평 단면)을 파먹는 과정"""
        
        viewmat, K = self._get_gsplat_camera_matrices(azimuth=0.0, elevation=90.0) 
        slice_centers, slice_thickness = self._get_horizontal_slice_centers(steps=70)
        
        pbar = tqdm(slice_centers[10:60])
        
        for i, center_pos in enumerate(pbar):
            
            pos = self.gaussians_save["means"]
            mask_suf = self._get_horizontal_slice_mask(pos, viewmat, center_pos, slice_thickness)
            
            means_cs = self.gaussians_save["means"][mask_suf]
            quats_cs = self.gaussians_save["quats"][mask_suf]
            scales_cs = self.gaussians_save["scales"][mask_suf]
            opacities_cs = self.gaussians_save["opacities"][mask_suf]
            shs_cs = self.gaussians_save["colors"][mask_suf]
            
            colors_precomp_cs = self._convert_SH_to_RGB(shs_cs)
            quats_activated = F.normalize(quats_cs, p=2, dim=-1)
            scales_activated = torch.exp(scales_cs)
            opacity_activated = torch.sigmoid(opacities_cs).squeeze(-1)
            
            # 3. 🌟 순수 3DGS 래스터라이제이션
            render_image, _, render_info = rasterization(
                means= means_cs,
                quats= quats_activated,                
                scales= scales_activated,             
                opacities= opacity_activated,
                colors= colors_precomp_cs,
                viewmats=viewmat.unsqueeze(0), 
                Ks=K.unsqueeze(0),             
                width=self.trainer_cfg.image_size,
                height=self.trainer_cfg.image_size,
                packed = True,
                backgrounds= self.background
            )
            
            # 4. SD 모델 추론 (두 번째 GPU 사용)
            render_image = render_image.squeeze(0)
            depth_map = F.interpolate(render_image.permute(2, 0, 1).unsqueeze(0), size=(384, 384), mode='bilinear')[0]
            depth_map = depth_map.unsqueeze(0).to(self.devices[-2])
            
            self._save_render_image(render_image, f"h{i}_render")
            
            with torch.no_grad():
                predicted_depth = self.pipe_horizontal.depth_estimator(depth_map).predicted_depth
                depth_resized = F.interpolate(predicted_depth.unsqueeze(0), size=(512, 512), mode='bilinear').squeeze(0)

            if epoch % self.trainer_cfg.sds_per_epoch == 0:
                ref_img = self._generate_sds_reference(render_image, step=self.trainer_cfg.sds_per_epoch - epoch//100, mode="horizontal")
                self._save_ref_image(ref_img, f"h{i}_ref")
            else:
                ref_img = self._load_cached_reference(f"h{i}_ref.png")
            
            ground_truth = self.transform(ref_img).to(self.device)
            loss = self.get_loss(render_image, ground_truth)
            
            # 5. gsplat 비서 보고 및 역전파!
            self.strategy.step_pre_backward(
                params=self.gaussians_save,
                optimizers=self.optimizer,
                state=self.strategy_state,
                step=self.global_step, 
                info=render_info,
            )
            
            render_info["means2d"].retain_grad()
            loss.backward()
            
            self.strategy.step_post_backward(
                params=self.gaussians_save,
                optimizers=self.optimizer,
                state=self.strategy_state,
                step=self.global_step,
                info=render_info,
                packed = True
            )
            
            for opt in self.optimizer.values():
                opt.step()
                opt.zero_grad()
            
            self.global_step += 1
            
            pbar.set_description(f"🔪 [수평 단면 뷰] {i}/50 속살 슬라이스 촬영 및 학습 시작! LOSS: {loss.item():.5f}")
            torch.cuda.empty_cache()
    
    
    def _regularize_with_original(self):
        """
        [Phase 3: 형태 유지 방어선]
        무작위 각도에서 원본 오렌지와 현재 오렌지를 비교하여, 
        학습 도중 오렌지의 전체적인 형태(Geometry)가 망가지는 것을 방지합니다.
        """
        print("🛡️ [형태 보존 뷰] 원본 가우시안과 형태 맞추기 (Regularization) 시작!")
        
        pbar = tqdm(range(30))
        
        for i in range(30):
            # 1. 무작위 카메라 앵글 생성 (원본 코드와 동일하게 방방곡곡에서 찍음!)
            rand_azimuth = random.uniform(0, 360)
            rand_elevation = random.uniform(-90, 90)
            viewmat, K = self._get_gsplat_camera_matrices(azimuth=rand_azimuth, elevation=rand_elevation) 
            
            # =========================================================
            # 2. [정답지] 원본 가우시안 렌더링 (no_grad로 메모리 철벽 방어!)
            # =========================================================
            with torch.no_grad():
                pos_orig = self.gaussians_orig["means"]
                shs_orig = self.gaussians_orig["colors"]
                opacity_orig = self.gaussians_orig["opacities"]
                scales_orig = self.gaussians_orig["scales"]
                quats_orig = self.gaussians_orig["quats"]
                
                colors_precomp_orig = self._convert_SH_to_RGB(shs_orig)
                scales_orig_act = torch.exp(scales_orig)
                quats_orig_act = F.normalize(quats_orig, p=2, dim=-1)
                opacity_orig_act = torch.sigmoid(opacity_orig).squeeze(-1)
                
                render_orig, _, _ = rasterization(
                    means=pos_orig,
                    quats=quats_orig_act,
                    scales=scales_orig_act,
                    opacities=opacity_orig_act,
                    colors=colors_precomp_orig,
                    viewmats=viewmat.unsqueeze(0),
                    Ks=K.unsqueeze(0),
                    width=self.trainer_cfg.image_size,
                    height=self.trainer_cfg.image_size,
                    packed = True,
                backgrounds= self.background
                )
                
                ground_truth = render_orig.detach() 
            
            
            pos = self.gaussians_save["means"]
            shs = self.gaussians_save["colors"]
            opacity = self.gaussians_save["opacities"]
            scales = self.gaussians_save["scales"]
            quats = self.gaussians_save["quats"]
            
            
            scales_act = torch.exp(scales)
            quats_act = F.normalize(quats, p=2, dim=-1)
            opacity_act = torch.sigmoid(opacity).squeeze(-1)
            colors_precomp = torch.clamp(self._convert_SH_to_RGB(shs), 0.0, 1.0)
            
            render_current, _, render_info = rasterization(
                means=pos,
                quats=quats_act,
                scales=scales_act,
                opacities=opacity_act,
                colors=colors_precomp,
                viewmats=viewmat.unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=self.trainer_cfg.image_size,
                height=self.trainer_cfg.image_size,
                packed = True,
                backgrounds= self.background
            )
            
            self._save_render_image(render_current, f"orig_render")
            
            
            loss = self.get_loss(render_current, ground_truth, ssim_weight=0.6)
            
            # 5. gsplat 비서 보고 및 역전파!
            self.strategy.step_pre_backward(
                params=self.gaussians_save,
                optimizers=self.optimizer,
                state=self.strategy_state,
                step=self.global_step, 
                info=render_info       
            )
            
            render_info["means2d"].retain_grad()
            loss.backward()
            
            self.strategy.step_post_backward(
                params=self.gaussians_save,
                optimizers=self.optimizer,
                state=self.strategy_state,
                step=self.global_step,
                info=render_info,
                packed = True
            )
            
            for opt in self.optimizer.values():
                opt.step()
                opt.zero_grad()
            
            # 🚨 잊지 말고 스텝 수 증가!
            self.global_step += 1
            torch.cuda.empty_cache()
            
            pbar.set_description(f"🛡️ [형태 보존 뷰] {i}/30 원본 가우시안과 형태 맞추기 시작! LOSS: {loss.item():.5f}")
    
    
    
    def _load_cached_reference(self, filename: str):
        """이전에 SD 모델이 그려준 정답지(Reference)를 디스크에서 불러옵니다."""
        # 출력 경로가 args나 config에 있다고 가정
        output_dir = self.trainer_cfg.output_path if hasattr(self.trainer_cfg, 'output_path') else "./output"
        filepath = os.path.join(output_dir, "output_ref", filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"🚨 There is no Cached Reference! {filepath}")
            
        return Image.open(filepath)
    
    
    
    def _save_render_image(self, render_image, view_name):
        
        output_dir = self.trainer_cfg.output_path if hasattr(self.trainer_cfg, 'output_path') else "./output"
        os.makedirs(os.path.join(output_dir, "output_renders"), exist_ok=True)
        
        if render_image.dim() == 4:
            render_image = render_image.squeeze(0)
        
        img_to_save = render_image.detach().cpu().permute(2, 0, 1)
        
        torchvision.utils.save_image(img_to_save, os.path.join(output_dir, "output_renders", f"{view_name}.png"))
    
    
    
    def _save_ref_image(self, ref_image, view_name):
        
        output_dir = self.trainer_cfg.output_path if hasattr(self.trainer_cfg, 'output_path') else "./output"
        os.makedirs(os.path.join(output_dir, "output_ref"), exist_ok=True)
        
        ref_image.save(os.path.join(output_dir, "output_ref", f"{view_name}.png"))
    
    
    
    def _save_gaussian_ply(self, epoch):
        
        output_dir = self.trainer_cfg.output_path if hasattr(self.trainer_cfg, 'output_path') else "./output"
        os.makedirs(os.path.join(output_dir, "output_models"), exist_ok=True)
        
        save_dict = {k: v.detach().cpu() for k, v in self.gaussians_save.items()}
        torch.save(save_dict, os.path.join(output_dir, "output_models", f"orange_ninja_epoch_{epoch}.pt"))
        
        print(f"💾 [저장 완료] Epoch {epoch} - 가우시안 모델이 안전하게 보관되었습니다!")
    
    
    
    def train(self):
        for epoch in range(self.trainer_cfg.epochs):
            print(f"------------------------- EPOCH {epoch} -------------------------")
            self.manage_densification_and_smoothing(epoch)

            self._train_vertical_views(epoch)
            self._train_horizontal_views(epoch)
            self._regularize_with_original()
            self._save_gaussian_ply(epoch)






@hydra.main(version_base=None, config_path="config", config_name="trainer_config")
def main(cfg: TrainerConfig):
    
    print("⚙️ [Hydra] 설정 파일 로드 완료!")
    
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    trainer_config = TrainerConfig(**config_dict)
    trainer = Trainer(trainer_config)
    trainer.train()


if __name__ == "__main__":
    main()