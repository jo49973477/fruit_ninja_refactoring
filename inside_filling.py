import os
import torch
import numpy as np
from plyfile import PlyData, PlyElement

# 🌟 [다이어트 포인트 1] Taichi를 부르긴 하지만, 메모리를 꽉 조여놓습니다!
import taichi as ti
import hydra
from omegaconf import DictConfig, OmegaConf

from utils.configs import FillingConfig
from utils.transformation import *
from utils.filling import fill_particles


class InternalFilling:
    
    
    
    def __init__(self, fill_config: FillingConfig):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fill_cfg = fill_config
        
        # 🌟 [다이어트 포인트 3] 16.0GB나 먹던 포크레인을 2.0GB 소형으로 교체!
        ti.init(arch=ti.cuda, device_memory_GB=16.0)
        print("🤖 삐리빅! 초경량 Taichi 엔진(2.0GB) 대기 완료!")
        
        self.gaussians = self.load_ply_to_gsplat(self.fill_cfg.model_path, self.device)
        self.rotation_matrices = self._setup_rotations()
        
        bg_np = [1, 1, 1] if self.fill_cfg.white_br else [0, 0, 0]
        self.background = torch.tensor(bg_np, device=self.device)
    
    
    
    def load_ply_to_gsplat(self, ply_path: str, device: str) -> dict:
        """PLY 파일을 읽어서 gsplat 딕셔너리로 변환 (범용 점 구름 호환 패치 적용!)"""
        print(f"📂 PLY 로딩 중: {ply_path}")
        
        plydata = PlyData.read(ply_path)
        vertex_data = plydata.elements[0].data
        
        # 1. xyz 좌표는 어떤 PLY든 무조건 공통!
        xyz = np.stack((np.asarray(vertex_data['x']),
                        np.asarray(vertex_data['y']),
                        np.asarray(vertex_data['z'])), axis=1)
        num_pts = xyz.shape[0]

        # 🚨 [로보코 센세의 하이브리드 분기점]
        if 'opacity' not in vertex_data.dtype.names:
            print("⚠️ [로보코 경고] 범용 점 구름(x,y,z,r,g,b) 감지! 임의의 미니 가우시안으로 둔갑시킵니다!")
            
            # Opacity: 완전히 불투명하게 (Sigmoid 역산을 고려해 10.0 대입)
            opacities = np.full((num_pts, 1), 10.0, dtype=np.float32)
            
            # Scale: 아주 작은 둥근 깍두기 모양 (3DGS는 log로 받으므로 exp(-4.6) ≒ 0.01)
            scales = np.full((num_pts, 3), -4.6, dtype=np.float32)
            
            # Quats (회전): 기본 형태 유지 (w=1, x=0, y=0, z=0)
            quats = np.zeros((num_pts, 4), dtype=np.float32)
            quats[:, 0] = 1.0
            
            # Colors: 범용 PLY의 red, green, blue(0~255)를 가져와서 SH 0차항으로 변환!
            rgb = np.stack((np.asarray(vertex_data['red']),
                            np.asarray(vertex_data['green']),
                            np.asarray(vertex_data['blue'])), axis=1) / 255.0
            colors = (rgb - 0.5) / 0.28209  
            
            # Features Extra: 0으로 텅 비워줌
            features_extra = np.zeros((num_pts, 45), dtype=np.float32)
            
        else:
            print("🌟 [로보코 감지] 오리지널 3DGS 형태의 PLY입니다! 정상 로드합니다.")
            # 캡틴의 원래 완벽한 로직 그대로!
            opacities = np.asarray(vertex_data['opacity'])[..., np.newaxis]
            scales = np.stack([np.asarray(vertex_data[f'scale_{i}']) for i in range(3)], axis=1)
            quats = np.stack([np.asarray(vertex_data[f'rot_{i}']) for i in range(4)], axis=1)
            colors = np.stack([np.asarray(vertex_data[f'f_dc_{i}']) for i in range(3)], axis=1)
            
            features_extra = np.zeros((num_pts, 45), dtype=np.float32)
            for i in range(45):
                if f'f_rest_{i}' in vertex_data.dtype.names:
                    features_extra[:, i] = np.asarray(vertex_data[f'f_rest_{i}'])

        # 2. 텐서 변환 (공통 로직)
        features_extra = torch.tensor(features_extra, dtype=torch.float32, device=device)
        means = torch.tensor(xyz, dtype=torch.float32, device=device)
        scales = torch.tensor(scales, dtype=torch.float32, device=device)
        quats = torch.tensor(quats, dtype=torch.float32, device=device)
        opacities = torch.tensor(opacities, dtype=torch.float32, device=device)
        colors = torch.tensor(colors, dtype=torch.float32, device=device)

        # 3. 사전 계산 (공통 로직)
        cov3D_precomp = build_cov3D_from_scales_quats(scales, quats)
        screen_points = torch.zeros((means.shape[0], 2), dtype=torch.float32, device=device, requires_grad=True)

        print(f"✅ {means.shape[0]}개의 껍질 가우시안 로드 완료!")

        return {
            "means": means,
            "scales": scales,
            "quats": quats,
            "opacities": opacities,
            "colors": colors,
            "cov3D_precomp": cov3D_precomp,   
            "screen_points": screen_points,
            "features_extra": features_extra,  
        }
        
        
        
    def _setup_rotations(self):
        """🌟 [다이어트 포인트 4] physics_cfg 대신 fill_cfg에서 값을 바로 가져옵니다!"""
        # FillingConfig에 rotation_degree와 rotation_axis를 추가했다고 가정!
        return generate_rotation_matrices(
            self.fill_cfg.rotation_degree, 
            self.fill_cfg.rotation_axis
        ).to(self.device)
        
        
        
    def execute(self) -> dict:
        """[Main Pipeline]"""
        print("🚀 내부 채우기 공정 시작!")
        
        # 1. 껍질 필터링 (🚨 스파이 완전 박멸 & 무적 모드 발동!)
        print("🤖 삐리빅! 껍질 점들을 100% 무적 모드로 통과시킵니다!")
        
        # mask라는 단어 자체를 쓰지 않습니다! 무조건 전체 복사!
        new_gaussians = {k: v.clone() for k, v in self.gaussians.items()}
        
        # 🌟 Taichi에 들어가기 직전, 인원 체크!
        survivors = new_gaussians["means"].shape[0]
        print(f"🍊 도마에 올라갈 준비가 된 껍질 점의 개수: {survivors}개!")
        
        if survivors == 0:
            raise ValueError("🚨 삐리빅! 원본 PLY 파일 자체가 텅 비어있습니다!")

        # 2. 회전 및 중앙 정렬 (도마 위로 올리기)
        new_gaussians["means"] = apply_rotations(new_gaussians["means"], self.rotation_matrices)
        new_gaussians["cov3D_precomp"] = apply_cov_rotations(new_gaussians["cov3D_precomp"], self.rotation_matrices)

        new_gaussians["means"], scale_origin, mean_pos = transform2origin(new_gaussians["means"])
        new_gaussians["means"] = shift2center111(new_gaussians["means"])
        new_gaussians["cov3D_precomp"] = scale_origin ** 2 * new_gaussians["cov3D_precomp"]
        
        # 3. Taichi 포크레인 출동!
        print(f"🤖 도마 정렬 완료! Taichi 포크레인으로 빈 공간을 채웁니다...")
        filled_pos = fill_particles(
            pos=new_gaussians["means"],
            opacity=new_gaussians["opacities"],
            cov=new_gaussians["cov3D_precomp"],
            grid_n=self.fill_cfg.grid_n, 
            grid_dx=1.0 / self.fill_cfg.grid_n,
            density_thres=self.fill_cfg.density_threshold,
            search_thres = 1,
            max_particles_per_cell=1,
            max_samples=19000000,
            search_exclude_dir=2,
            ray_cast_dir=3,
            boundary=[
                0.51,
                1.50,
                0.57,
                1.43,
                0.51,
                1.49
            ],
            smooth=False,
        ).to(self.device)
        
        init_num = new_gaussians["means"].shape[0]
        new_points_normalized = filled_pos[init_num:].clone()
        
        # 도마 위에서 1x1x1로 찌그러뜨렸던 걸 원래 크기와 위치로 뻥! 튀깁니다.
        new_points = undoshift2center111(new_points_normalized)
        new_points = undotransform2origin(new_points, scale_origin, mean_pos)
        
        # 회전시켰던 것도 반대 방향으로 휙! 돌려놓습니다.
        new_points = apply_inverse_rotations(new_points, self.rotation_matrices)
        
        # 새로 태어난 속살 점들의 총 개수
        num_new = new_points.shape[0]
        print(f"🍊 싱싱한 오렌지 속살 점 {num_new}개 추출 완료!")


        # ---------------------------------------------------------
        # 🌟 Step 5: 기존 가우시안 딕셔너리와 합치기 (생명 불어넣기)
        # ---------------------------------------------------------
        print("🧬 속살 점들에게 가우시안 생명(크기, 색상)을 부여하고 껍질과 융합합니다!")
        
        # 1. 스케일 (크기): 속살은 빈틈없이 꽉 채워야 하니까 적당히 작게 설정합니다.
        # (보통 가우시안 스케일은 exp()로 계산되므로 음수값을 줍니다. ex: -5.0)
        new_scales = torch.ones((num_new, 3), dtype=torch.float32, device=self.device) * -5.0
        
        # 2. 회전 (Quaternions): 기본 회전값 [1, 0, 0, 0]을 줍니다. 동그란 점이니까요!
        new_quats = torch.zeros((num_new, 4), dtype=torch.float32, device=self.device)
        new_quats[:, 0] = 1.0
        
        # 3. 투명도 (Opacity): 속살이니까 텅 비어 보이면 안 되죠! 1.0(완전 불투명)으로 설정!
        # (sigmoid를 통과하기 전의 logit 값이므로 아주 큰 값인 100.0을 줍니다)
        new_opacities = torch.ones((num_new, 1), dtype=torch.float32, device=self.device) * 1.386
        
        # 4. 색상 (Colors): 일단 임시로 아주 예쁜 귤색(주황) 베이스를 발라줍니다!
        # 나중에 SDS 엔진이 이 색을 바탕으로 진짜 과육처럼 깎아줄 거예요. RGB [1.0, 0.5, 0.0]
        base_color = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32, device=self.device)
        new_colors = base_color.repeat(num_new, 1)

        # 5. cov3D_precomp 계산 (gsplat 렌더링에 필요한 공분산 행렬)
        new_cov3D_precomp = build_cov3D_from_scales_quats(new_scales, new_quats)
        new_screen_points = torch.zeros((num_new, 2), dtype=torch.float32, device=self.device, requires_grad=True)
        
        new_features_extra = torch.zeros((num_new, 45), dtype=torch.float32, device=self.device)

        # 🌟 대망의 융합! (원본 껍질 self.gaussians + 새로운 속살)
        final_gaussians = {
            "means": torch.cat([self.gaussians["means"], new_points], dim=0),
            "scales": torch.cat([self.gaussians["scales"], new_scales], dim=0),
            "quats": torch.cat([self.gaussians["quats"], new_quats], dim=0),
            "opacities": torch.cat([self.gaussians["opacities"], new_opacities], dim=0),
            "colors": torch.cat([self.gaussians["colors"], new_colors], dim=0),
            "features_extra": torch.cat([self.gaussians["features_extra"], new_features_extra], dim=0),
            "cov3D_precomp": torch.cat([self.gaussians["cov3D_precomp"], new_cov3D_precomp], dim=0),
            "screen_points": torch.cat([self.gaussians["screen_points"], new_screen_points], dim=0),
        }

        print("✨ 공정 완료! 다이어트된 엔진으로 오렌지 속이 완벽하게 채워졌습니다!")
        
        return final_gaussians
    
    def save_gsplat_to_ply(self, gaussians: dict):
        """
        꽉 채워진 가우시안 딕셔너리를 표준 3DGS .ply 파일로 구워냅니다!
        """
        
        print(f"🤖 삐리빅! 오렌지를 [{self.fill_cfg.output_path}]에 굽기 시작합니다...")

        # 1. GPU에 있는 텐서들을 CPU로 데려와서 Numpy로 변환!
        means = gaussians["means"].detach().cpu().numpy()
        scales = gaussians["scales"].detach().cpu().numpy()
        quats = gaussians["quats"].detach().cpu().numpy()
        opacities = gaussians["opacities"].detach().cpu().numpy()
        colors = gaussians["colors"].detach().cpu().numpy()
        features_extra = gaussians["features_extra"].detach().cpu().numpy()

        # 법선 벡터(Normal)는 3DGS 기본 포맷을 맞추기 위해 0으로 채워줍니다.
        normals = np.zeros_like(means)

        # 2. PLY 파일이 요구하는 데이터 타입(구조체) 정의
        dtype_base = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ]
        dtype_f_rest = [(f'f_rest_{i}', 'f4') for i in range(45)] # 45개의 SH 항목
        dtype_tail = [
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
        ]
        
        dtype_full = dtype_base + dtype_f_rest + dtype_tail
        
        elements = np.empty(means.shape[0], dtype=dtype_full)
        
        for i in range(45):
            elements[f'f_rest_{i}'] = features_extra[:, i]
        
        elements['x'], elements['y'], elements['z'] = means[:, 0], means[:, 1], means[:, 2]
        elements['nx'], elements['ny'], elements['nz'] = normals[:, 0], normals[:, 1], normals[:, 2]
        
        elements['f_dc_0'], elements['f_dc_1'], elements['f_dc_2'] = colors[:, 0], colors[:, 1], colors[:, 2]
        elements['opacity'] = opacities[:, 0]
        
        elements['scale_0'], elements['scale_1'], elements['scale_2'] = scales[:, 0], scales[:, 1], scales[:, 2]
        elements['rot_0'], elements['rot_1'], elements['rot_2'], elements['rot_3'] = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

        # 4. PLY 요소로 만들어서 파일로 쓰기!
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(self.fill_cfg.output_path)
        
        print(f"✨ 굽기 완료! 이제 이 .ply 파일을 렌더러에 바로 던져주면 됩니다!")

@hydra.main(version_base=None, config_path="config", config_name="filling_config")
def main(cfg: FillingConfig):
    
    print("⚙️ [Hydra] 설정 파일 로드 완료!")
    
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    filler_config = FillingConfig(**config_dict)
    filler = InternalFilling(filler_config)
    gaussian = filler.execute()
    filler.save_gsplat_to_ply(gaussian)


if __name__ == "__main__":
    main()
