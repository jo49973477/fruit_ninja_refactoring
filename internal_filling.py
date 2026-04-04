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

from utils.configs import TrainerConfig, PhysicsConfig, FillingConfig
from utils.transformation import *


class InternalFilling:
    def __init__(self, fill_config: FillingConfig, physics_config: PhysicsConfig):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.fill_cfg = fill_config
        self.physics_cfg = physics_config
        
        self.gaussians = self.load_ply_to_gsplat(self.fill_cfg.model_path, self.device)
        self.rotation_matrices = self._setup_rotations()
        
        bg_np = [1, 1, 1] if self.fill_cfg.white_br else [0, 0, 0]
        self.background = torch.tensor(bg_np, device=self.device)
    
    
    
    def load_ply_to_gsplat(self, ply_path: str, device: str) -> dict:
        """PLY 파일을 읽어서 gsplat 및 물리 엔진용 딕셔너리로 변환합니다."""
        print(f"📂 PLY 로딩 중: {ply_path}")
        
        plydata = PlyData.read(ply_path)
        xyz = np.stack((np.asarray(plydata.element['vertex']['x']),
                        np.asarray(plydata.element['vertex']['y']),
                        np.asarray(plydata.element['vertex']['z'])), axis=1)
        
        opacities = np.asarray(plydata.element['vertex']['opacity'])[..., np.newaxis]
        
        # 스케일과 회전 (quats) 가져오기
        scales = np.stack([np.asarray(plydata.element['vertex'][f'scale_{i}']) for i in range(3)], axis=1)
        quats = np.stack([np.asarray(plydata.element['vertex'][f'rot_{i}']) for i in range(4)], axis=1)
        
        # 색상 (SH0 성분만 가져오기 - f_dc_0, 1, 2)
        colors = np.stack([np.asarray(plydata.element['vertex'][f'f_dc_{i}']) for i in range(3)], axis=1)

        # --- 텐서 변환 ---
        means = torch.tensor(xyz, dtype=torch.float32, device=device)
        scales = torch.tensor(scales, dtype=torch.float32, device=device)
        quats = torch.tensor(quats, dtype=torch.float32, device=device)
        opacities = torch.tensor(opacities, dtype=torch.float32, device=device)
        colors = torch.tensor(colors, dtype=torch.float32, device=device)

        cov3D_precomp = build_cov3D_from_scales_quats(scales, quats)

        screen_points = torch.zeros((means.shape[0], 2), dtype=torch.float32, device=device, requires_grad=True)

        print(f"✅ {means.shape[0]}개의 가우시안 로드 완료!")

        return {
            "means": means,
            "scales": scales,
            "quats": quats,
            "opacities": opacities,
            "colors": colors,
            "cov3D_precomp": cov3D_precomp,   # [N, 6]
            "screen_points": screen_points    # [N, 2]
        }
    
    
        
    def _setup_rotations(self):
        """회전 행렬 등 변환에 필요한 수학적 도구 세팅"""
        return generate_rotation_matrices(self.physics_cfg.preprocessing_params.rotation_degree, 
                                                  self.physics_cfg.preprocessing_params.rotation_axis).to(self.device)

    def _to_mpm_space(self, data: dict):
        """가우시안들을 물리 엔진용 도마(MPM Space) 위로 올리기"""
        # transform2origin, shift2center111 등 적용
        # return mpm_pos, mpm_cov, ...
        pass

    def _from_mpm_space(self, mpm_pos: torch.Tensor, scale_origin, mean_pos):
        """생성된 속살 입자들을 다시 원래 세계(World Space)로 복원"""
        # undotransform2origin, undoshift2center111 등 적용
        # return world_pos
        pass

    # --- 🚀 핵심 실행 메서드 (Public) ---

    def execute(self) -> dict:
        """
        [Main Pipeline]
        이 메서드 하나만 호출하면 내부 채우기가 완료된 최종 딕셔너리를 뱉어냅니다!
        """
        
        print("🚀 내부 채우기 공정 시작!")
        
        mpm_data = self._to_mpm_space(self.gaussians)
        
        mask = self.gaussians["opacities"][:, 0] > self.physics_cfg.preprocessing_params.opacity_threshold
        
        new_gaussians = {k: v[mask] for k, v in self.gaussians.items()}
        
        new_gaussians["pos"] = apply_rotations(new_gaussians["pos"], self.rotation_matrices)
        new_gaussians["cov3D_precomp"] = apply_rotations(new_gaussians["cov3D_precomp"], self.rotation_matrices)

        new_gaussians["pos"], scale_origin, mean_pos = transform2origin(new_gaussians)
        new_gaussians["pos"] =  shift2center111(new_gaussians["pos"])
        new_gaussians["cov3D_precomp"] = scale_origin ** 2 * new_gaussians["cov3D_precomp"]
        
        
        
        # Step 4: 새로 만들어진 점들만 골라내서 원래 세계로 복구
        # new_means_world = self._from_mpm_space(filled_pos[original_size:], ...)

        # Step 5: 기존 가우시안 딕셔너리와 합치기
        # final_gaussians = self._merge(self.gaussians, new_means_world)

        print("✨ 공정 완료! 이제 오렌지 속이 꽉 찼습니다!")
        return final_gaussians

    def save_result(self, data: dict, path: str):
        """결과물을 PLY 파일로 저장 (확인용)"""
        pass