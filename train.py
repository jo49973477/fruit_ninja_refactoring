import os
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from plyfile import PlyData
import numpy as np
from tqdm import tqdm

from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from diffusers import StableDiffusionDepth2ImgPipeline

from utils.configs import TrainerConfig, PhysicsConfig
from utils.transformation import *


class Trainer:
    
    
    
    def __init__(self, trainer_config: TrainerConfig, physics_config: PhysicsConfig):
        self.devices = self._get_optimal_devices()
        self.device = self.devices[0]
        
        self.trainer_cfg = trainer_config
        self.physics_cfg = physics_config
        self.background = torch.tensor([1, 1, 1], dtype=torch.float32).to(self.device)
        
        self.gaussians_save = self.load_ply_to_gsplat(self.trainer_cfg.gaussian_path, self.device)
        self.gaussians_orig = self.load_ply_to_gsplat(self.trainer_cfg.gaussian_orig, self.device)
        
        self.rotation_matrices = generate_rotation_matrix(degree = self.physics_cfg.preprocessing_params.rotation_degree,
                                                          axis = self.physics_cfg.preprocessing_params.rotation_axis).to(self.device)
        
        self.mpm_space_viewpoint_center = torch.tensor(self.physics_cfg.camera_params.mpm_space_viewpoint_center).reshape(1, 3).to(self.device)
        self.mpm_space_vertical_upward_axis = torch.tensor(self.physics_cfg.camera_params.mpm_space_vertical_upward_axis).reshape(1, 3).to(self.device)
        
        self.transform = T.ToTensor()
        
        
        dev_vertical, dev_horizontal = self.devices[-2], self.devices[-1]
        
        try:
            print(f"🚀 수직(Vertical) SD 모델을 {dev_vertical}에 욱여넣는 중...")
            self.pipe_vertical = StableDiffusionDepth2ImgPipeline.from_pretrained(
                self.trainer_cfg.sd_model_vertical
            ).to(dev_vertical)
        except Exception as e:
            # 3. 진짜 에러 원인({e})을 캡틴에게 낱낱이 보고하라!!
            raise RuntimeError(
                f"\n🚨 수직 SD 모델({self.trainer_cfg.sd_model_vertical}) 로딩 실패!\n"
                f"단순히 모델이 없는 게 아니라, VRAM 부족이나 인터넷 끊김일 수도 있어!\n"
                f"🔥 범인(진짜 원인): {e}"
            ) from e # <== 파이썬에게 "원래 에러 흔적 남겨둬!" 라고 지시하는 마법의 키워드

        try:
            print(f"🚀 수평(Horizontal) SD 모델을 {dev_horizontal}에 욱여넣는 중...")
            self.pipe_horizontal = StableDiffusionDepth2ImgPipeline.from_pretrained(
                self.trainer_cfg.sd_model_horizontal
            ).to(dev_horizontal)
        except Exception as e:
            raise RuntimeError(
                f"\n🚨 수평 SD 모델({self.trainer_cfg.sd_model_horizontal}) 로딩 실패!\n"
                f"🔥 범인(진짜 원인): {e}"
            ) from e
            
        
        self.strategy = DefaultStrategy(
            refine_start_iter=100,
            refine_stop_iter=1500,
            verbose=True
        )
        
        self.strategy_state = self.strategy.initialize_state(self.gaussians_save, self.device)
    
    
    
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
        for idx in tqdm(range(xyz.size(0)), desc="Creating 3D Grid"):
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

        for cell_key, indices in tqdm(grid.items(), desc="Smoothing Gaussians in Grid"):
            if len(indices) > 0:
                cell_features = features[indices]
                avg_features = torch.mean(cell_features, dim=0).squeeze(0)
                smoothed_features[indices] = avg_features
                counts[indices] += 1
        
        with torch.no_grad():
            gaussians_dict["colors"].copy_(smoothed_features)



    def preprocess_particles(self, gaussians_dict: dict):
        """Converting gaussian for MPM physical engine."""
        
        init_pos = gaussians_dict["means"]
        init_opacity = gaussians_dict["opacities"]
        init_shs = gaussians_dict["colors"]
        
        init_cov = build_cov3D_from_scales_quats(gaussians_dict["scales"], gaussians_dict["quats"])

        # 2. 글로벌 변수 극혐! 캡틴의 클래스 변수(self.*)를 당당하게 사용!
        transformed_pos, scale_origin, original_mean_pos = transform2origin(init_pos)
        transformed_pos = shift2center111(transformed_pos)

        # Modify covariance matrix accordingly
        init_cov = apply_cov_rotations(init_cov, self.rotation_matrices) # self.rotation_matrices 사용!
        init_cov = scale_origin * scale_origin * init_cov

        mpm_init_pos = transformed_pos.to(device=self.device)
        
        return init_shs, init_opacity, mpm_init_pos, init_cov, scale_origin, original_mean_pos
    
    
    
    def manage_densification_and_smoothing(self, epoch: int):
        """매 에폭마다 호출되어 가우시안 증식(Densify)과 스무딩을 관리합니다."""
        
        # --- 1. Densification & Pruning (10 에폭마다) ---
        if epoch > 1 and epoch % 10 == 0:
            print(f"\n🚀 [Epoch {epoch}] 증식(Densify) 전 가우시안 개수: {self.gaussians_save['means'].shape[0]}")
            
            self.strategy.step_post_backward(
                params=self.gaussians_save,        # 우리의 가우시안 딕셔너리
                optimizers={"gaussians": self.optimizer}, # 옵티마이저 (Adam 상태 복제용)
                strategy_state=self.strategy_state, # 비서의 수첩
                step=epoch,                        # 현재 에폭(스텝)
                info=self.last_render_info,        # 렌더링할 때 받았던 그래디언트 정보! (중요)
                absgrad_threshold=0.0002,          
                min_opacity=0.005,
                extent=self.physics_cfg.preprocessing_params.extent # 오렌지 크기
            )

        # --- 2. Grid Smoothing (101 에폭마다) ---
        if epoch > 0 and epoch % 101 == 0:
            print(f"\n🌟 [Epoch {epoch}] 3D 격자 스무딩(Smoothing) 작업 가동!")
            
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
        radius = self.physics_cfg.camera_params.init_radius
        
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
        right = torch.cross(up, forward)
        right = right / torch.norm(right)
        up = torch.cross(forward, right)
        
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
    
    
    
    def _get_mask(self, pos: torch.Tensor, viewmat: torch.Tensor, vertical = True):
        """수직 뷰용 마스크 (겉껍질 렌더링이므로 기본적으로 다 살립니다!)"""
        # (나중에 화면 밖에 나가는 점들만 걸러내는 최적화를 추가할 수 있음)
        return torch.ones(pos.shape[0], dtype=torch.bool, device=self.device)
    
    
    
    def _convert_SH_to_RGB(self, shs: torch.Tensor,):
        """SH degree 0 텐서를 [0~1] 사이의 진짜 RGB 색상으로 변환합니다!"""
        # SH0의 마법의 상수 (수학적으로 정해진 값!)
        SH_C0 = 0.28209479177387814
        
        # 색상 계산: RGB = SH * C0 + 0.5
        colors = shs * SH_C0 + 0.5
        
        # RGB 값이 0 이하로 가거나 1을 넘지 않게 꽉 잡아주기(Clamp)
        return torch.clamp(colors, min=0.0, max=1.0)
    
         
    
    def _load_cached_reference(self, filename: str):
        """이전에 SD 모델이 그려준 정답지(Reference)를 디스크에서 불러옵니다."""
        # 출력 경로가 args나 config에 있다고 가정
        output_dir = self.trainer_cfg.output_path if hasattr(self.trainer_cfg, 'output_path') else "./output"
        filepath = os.path.join(output_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"🚨 캡틴! 캐시된 레퍼런스 이미지가 없어! 경로: {filepath}")
            
        return Image.open(filepath)
    
         
    
    def _train_vertical_views(self, epoch: int):
        
        for i in range(30):
            print(f"🎥 [수직 뷰] {i}/30 앵글 촬영 및 학습 시작!")
            
            init_shs, init_opacity, pos, cov3D, scale_origin, original_mean_pos = self.preprocess_particles(self.gaussians_save)
            
            viewmat, K = self._get_gsplat_camera_matrices(i * 12, elevation=0) 
            
            mask = self._get_mask(pos, viewmat) 
            
            pos_cs = pos[mask]
            cov3D_cs = cov3D[mask]
            shs_cs = init_shs[mask]
            opacity_cs = init_opacity[mask]
            
            colors_precomp_cs = self._convert_SH_to_RGB(shs_cs)
            
            render_image, render_alpha, render_info = rasterization(
                means=pos_cs,
                quats=None, # cov3D를 직접 넘기므로 quat, scale은 None!
                scales=None,
                opacities=opacity_cs,
                colors=colors_precomp_cs,
                viewmats=viewmat.unsqueeze(0), # [1, 4, 4] 형태로 전달
                Ks=K.unsqueeze(0),             # [1, 3, 3] 형태로 전달
                width= self.trainer_cfg.image_size,
                height= self.trainer_cfg.image_size,
                cov3Ds_precomp=cov3D_cs        # 우리가 만든 공분산 행렬 투척!
            )
            # =========================================================
            
            # 4. SD 모델로 뎁스(Depth) 뽑고 피드백 받기
            # (아까 우리가 고쳐놓은 self.pipe_vertical 디바이스로 보냄!)
            depth_map = F.interpolate(render_image.permute(2, 0, 1).unsqueeze(0), size=(384, 384), mode='bilinear')[0]
            depth_map = depth_map.unsqueeze(0).to(self.devices[-1]) # 안전하게 할당된 디바이스 사용!
            
            # SD 뎁스 추정 (requires_grad 끄는 거 잊지 말기!)
            with torch.no_grad():
                predicted_depth = self.pipe_vertical.depth_estimator(depth_map).predicted_depth
                depth_resized = F.interpolate(predicted_depth.unsqueeze(0), size=(512, 512), mode='bilinear').squeeze(0)

            # 5. 레퍼런스 이미지 생성 (또는 캐시에서 로드)
            if epoch % 30 == 0:
                ref_img = self._generate_sds_reference(render_image, depth_resized, step=30 - epoch//100, mode="vertical")
                # 이미지 저장 로직 등...
            else:
                ref_img = self._load_cached_reference(f"v{i}_ref.png")
            
            ground_truth = self.transform(ref_img).to(self.device)
            
            # 6. Loss 계산 및 역전파! (옵티마이저는 나중에 세팅할 예정)
            loss = 0.7 * self.get_ssim_loss(render_image, ground_truth) + 0.3 * F.mse_loss(render_image, ground_truth)
            
            # 옵티마이저 초기화 및 업데이트!
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # (오리지널 코드의 visibility_filter 업데이트나 모멘텀 제어는 나중에 gsplat 전략(Strategy) 객체로 처리!)
            
            torch.cuda.empty_cache() # VRAM 찌꺼기 청소!
    
    

    def _train_horizontal_views(self, epoch: int):
        """Phase 2: 카메라를 밀어 넣으며 가우시안의 속살(수평 단면)을 파먹는 과정"""
        
        # 1. 탑다운(위에서 아래로 보는) 카메라 세팅 (Azimuth=0, Elevation=90)
        # (수평 단면을 보려면 위에서 내려다봐야 함!)
        viewmat, K = self._get_gsplat_camera_matrices(azimuth=0.0, elevation=90.0) 
        
        # 2. 오렌지 중앙을 관통하며 썰어낼 50개의 칼날 위치(Centers) 계산
        # (오리지널 코드의 interpolate_along_camera_direction 대체 헬퍼 함수)
        slice_centers, slice_thickness = self._get_horizontal_slice_centers(steps=70)
        
        # 10번째부터 60번째까지만 썰기 (너무 겉이나 바닥은 패스!)
        for i, center_pos in enumerate(slice_centers[10:60]):
            print(f"🔪 [수평 단면 뷰] {i}/50 속살 슬라이스 촬영 및 학습 시작!")
            
            # 3. 전처리 (이건 이제 눈 감고도 하지!)
            init_shs, init_opacity, pos, cov3D, _, _ = self.preprocess_particles(self.gaussians_save)
            
            # 4. 🔪 프루트 닌자 컷!! (현재 높이 center_pos 기준으로 단면 자르기)
            # (오리지널의 plane_filter 로직 - mask_suf에 해당하는 부분)
            mask_suf = self._get_horizontal_slice_mask(pos, viewmat, center_pos, slice_thickness)
            
            pos_cs = pos[mask_suf]
            cov3D_cs = cov3D[mask_suf]
            shs_cs = init_shs[mask_suf]
            opacity_cs = init_opacity[mask_suf]
            
            colors_precomp_cs = self._convert_SH_to_RGB(shs_cs, viewmat, pos_cs)
            
            render_image, _, _ = rasterization(
                means=pos_cs,
                quats=None, 
                scales=None,
                opacities=opacity_cs,
                colors=colors_precomp_cs,
                viewmats=viewmat.unsqueeze(0), 
                Ks=K.unsqueeze(0),             
                width=self.trainer_cfg.image_size,
                height=self.trainer_cfg.image_size,
                cov3Ds_precomp=cov3D_cs        
            )
            # =========================================================
            
            # 5. 수평(Horizontal) SD 모델로 뎁스 뽑고 피드백 받기
            # 🚨 주의: 디바이스는 우리가 동적 할당한 self.devices[-2] (두 번째 서브 GPU)
            depth_map = F.interpolate(render_image.permute(2, 0, 1).unsqueeze(0), size=(384, 384), mode='bilinear')[0]
            depth_map = depth_map.unsqueeze(0).to(self.devices[-2]) 
            
            with torch.no_grad():
                # pipe_h(수평 모델) 사용!
                predicted_depth = self.pipe_horizontal.depth_estimator(depth_map).predicted_depth
                depth_resized = F.interpolate(predicted_depth.unsqueeze(0), size=(512, 512), mode='bilinear').squeeze(0)

            # 6. 레퍼런스 이미지(과육 텍스처) 생성 및 Loss 계산
            if epoch % 30 == 0:
                ref_img = self._generate_sds_reference(render_image, depth_resized, step=30 - epoch//100, mode="horizontal")
            else:
                ref_img = self._load_cached_reference(f"h{i}_ref.png")
            
            ground_truth = self.transform(ref_img).to(self.device)
            
            loss = 0.7 * self.get_ssim_loss(render_image, ground_truth) + 0.3 * F.mse_loss(render_image, ground_truth)
            
            # 7. 업데이트! "속살을 더 오렌지 과육처럼 만들어!!"
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            torch.cuda.empty_cache()
            
            
    
    def train(self):
        for epoch in range(self.trainer_cfg.epochs):
            self.manage_densification_and_smoothing(epoch)

            self._train_vertical_views(epoch)
            self._train_horizontal_views(epoch)
            self._regularize_with_original()