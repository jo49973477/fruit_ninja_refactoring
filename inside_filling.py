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

        ti.init(arch=ti.cuda, device_memory_GB=16.0)
        print("🤖Taichi Engine is ready!")

        self.gaussians = self.load_ply_to_gsplat(self.fill_cfg.model_path, self.device)
        self.rotation_matrices = self._setup_rotations()

        bg_np = [1, 1, 1] if self.fill_cfg.white_br else [0, 0, 0]
        self.background = torch.tensor(bg_np, device=self.device)

    def load_ply_to_gsplat(self, ply_path: str, device: str) -> dict:
        """Reading the ply file and converting it into Gaussian"""
        
        print(f"📂 Loading PLY files...: {ply_path}")

        plydata = PlyData.read(ply_path)
        vertex_data = plydata.elements[0].data

        xyz = np.stack(
            (
                np.asarray(vertex_data["x"]),
                np.asarray(vertex_data["y"]),
                np.asarray(vertex_data["z"]),
            ),
            axis=1,
        )
        num_pts = xyz.shape[0]

        if "opacity" not in vertex_data.dtype.names:
            print(
                "⚠️ [WARNING] Got the point cloud! (x,y,z,r,g,b) They will be changed into mini gaussians!"
            )

            opacities = np.full((num_pts, 1), 10.0, dtype=np.float32)

            scales = np.full((num_pts, 3), -4.6, dtype=np.float32)

            quats = np.zeros((num_pts, 4), dtype=np.float32)
            quats[:, 0] = 1.0

            rgb = (
                np.stack(
                    (
                        np.asarray(vertex_data["red"]),
                        np.asarray(vertex_data["green"]),
                        np.asarray(vertex_data["blue"]),
                    ),
                    axis=1,
                )
                / 255.0
            )
            colors = (rgb - 0.5) / 0.28209

            features_extra = np.zeros((num_pts, 45), dtype=np.float32)

        else:
            print("🌟 Original 3DGS formed-ply file! The normal loading will be processed!")
            
            opacities = np.asarray(vertex_data["opacity"])[..., np.newaxis]
            scales = np.stack(
                [np.asarray(vertex_data[f"scale_{i}"]) for i in range(3)], axis=1
            )
            quats = np.stack(
                [np.asarray(vertex_data[f"rot_{i}"]) for i in range(4)], axis=1
            )
            colors = np.stack(
                [np.asarray(vertex_data[f"f_dc_{i}"]) for i in range(3)], axis=1
            )

            features_extra = np.zeros((num_pts, 45), dtype=np.float32)
            for i in range(45):
                if f"f_rest_{i}" in vertex_data.dtype.names:
                    features_extra[:, i] = np.asarray(vertex_data[f"f_rest_{i}"])

        # 2. Converting it into Tensor
        features_extra = torch.tensor(
            features_extra, dtype=torch.float32, device=device
        )
        means = torch.tensor(xyz, dtype=torch.float32, device=device)
        scales = torch.tensor(scales, dtype=torch.float32, device=device)
        quats = torch.tensor(quats, dtype=torch.float32, device=device)
        opacities = torch.tensor(opacities, dtype=torch.float32, device=device)
        colors = torch.tensor(colors, dtype=torch.float32, device=device)

        # 3. prior computing
        cov3D_precomp = build_cov3D_from_scales_quats(scales, quats)
        screen_points = torch.zeros(
            (means.shape[0], 2), dtype=torch.float32, device=device, requires_grad=True
        )

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
        """setting up 'rotation degree' and 'rotation axis'!"""
        
        return generate_rotation_matrices(
            self.fill_cfg.rotation_degree, self.fill_cfg.rotation_axis
        ).to(self.device)

    def execute(self) -> dict:
        """[Main Pipeline]"""
        
        print("🚀 Internal filling starting....!")

        new_gaussians = {k: v.clone() for k, v in self.gaussians.items()}

        survivors = new_gaussians["means"].shape[0]
        print(f"🍊 {survivors} dots are available now!")

        if survivors == 0:
            raise ValueError("🚨 PLY FILE IS EMPTY! Ply File is empty")

        new_gaussians["means"] = apply_rotations(
            new_gaussians["means"], self.rotation_matrices
        )
        new_gaussians["cov3D_precomp"] = apply_cov_rotations(
            new_gaussians["cov3D_precomp"], self.rotation_matrices
        )

        new_gaussians["means"], scale_origin, mean_pos = transform2origin(
            new_gaussians["means"]
        )
        new_gaussians["means"] = shift2center111(new_gaussians["means"])
        new_gaussians["cov3D_precomp"] = (
            scale_origin**2 * new_gaussians["cov3D_precomp"]
        )

        print(f"🤖 Filling free spaces using taichi...")
        
        filled_pos = fill_particles(
            pos=new_gaussians["means"],
            opacity=new_gaussians["opacities"],
            cov=new_gaussians["cov3D_precomp"],
            
            # 1. Grid & Density Settings
            grid_n=self.fill_cfg.particle_params.n_grid,
            grid_dx=1.0 / self.fill_cfg.particle_params.n_grid,
            density_thres=self.fill_cfg.particle_params.density_threshold,
            
            # 2. Search & Ray Casting Settings 
            search_thres=self.fill_cfg.particle_params.search_threshold,
            max_particles_per_cell=self.fill_cfg.particle_params.max_partciels_per_cell,  # (원작자의 partciels 오타 유지)
            max_samples=self.fill_cfg.particle_params.max_particles_num,
            search_exclude_dir=self.fill_cfg.particle_params.search_exclude_direction,
            ray_cast_dir=self.fill_cfg.particle_params.ray_cast_direction,
            
            # 3. Geometry Limits
            boundary=self.fill_cfg.particle_params.boundary,
            smooth=self.fill_cfg.particle_params.smooth,
        ).to(self.device)

        init_num = new_gaussians["means"].shape[0]
        new_points_normalized = filled_pos[init_num:].clone()

        new_points = undoshift2center111(new_points_normalized)
        new_points = undotransform2origin(new_points, scale_origin, mean_pos)

        new_points = apply_inverse_rotations(new_points, self.rotation_matrices)

        num_new = new_points.shape[0]
        print(f"🍊 {num_new} filled internal parts are extracted!")

        print(
            "🧬 Giving side and color to internal dots and combining it with surface!"
        )

        new_scales = (
            torch.ones((num_new, 3), dtype=torch.float32, device=self.device) * -5.0
        )

        new_quats = torch.zeros((num_new, 4), dtype=torch.float32, device=self.device)
        new_quats[:, 0] = 1.0

        new_opacities = (
            torch.ones((num_new, 1), dtype=torch.float32, device=self.device) * 1.386
        )

        base_color = torch.tensor(
            [1.0, 0.5, 0.0], dtype=torch.float32, device=self.device
        )
        new_colors = base_color.repeat(num_new, 1)

        new_cov3D_precomp = build_cov3D_from_scales_quats(new_scales, new_quats)
        new_screen_points = torch.zeros(
            (num_new, 2), dtype=torch.float32, device=self.device, requires_grad=True
        )

        new_features_extra = torch.zeros(
            (num_new, 45), dtype=torch.float32, device=self.device
        )

        final_gaussians = {
            "means": torch.cat([self.gaussians["means"], new_points], dim=0),
            "scales": torch.cat([self.gaussians["scales"], new_scales], dim=0),
            "quats": torch.cat([self.gaussians["quats"], new_quats], dim=0),
            "opacities": torch.cat([self.gaussians["opacities"], new_opacities], dim=0),
            "colors": torch.cat([self.gaussians["colors"], new_colors], dim=0),
            "features_extra": torch.cat(
                [self.gaussians["features_extra"], new_features_extra], dim=0
            ),
            "cov3D_precomp": torch.cat(
                [self.gaussians["cov3D_precomp"], new_cov3D_precomp], dim=0
            ),
            "screen_points": torch.cat(
                [self.gaussians["screen_points"], new_screen_points], dim=0
            ),
        }

        print("✨ It's done! The inside of the product is filled completely!")

        return final_gaussians

    def save_gsplat_to_ply(self, gaussians: dict):
        """
        Converting Gaussian dictionary into standart 3DGS ply file!
        """

        print(f"🤖 3DGS dictionary will be converted into ply file, and saved in the [{self.fill_cfg.output_path}]...")

        means = gaussians["means"].detach().cpu().numpy()
        scales = gaussians["scales"].detach().cpu().numpy()
        quats = gaussians["quats"].detach().cpu().numpy()
        opacities = gaussians["opacities"].detach().cpu().numpy()
        colors = gaussians["colors"].detach().cpu().numpy()
        features_extra = gaussians["features_extra"].detach().cpu().numpy()

        normals = np.zeros_like(means)

        dtype_base = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("f_dc_0", "f4"),
            ("f_dc_1", "f4"),
            ("f_dc_2", "f4"),
        ]
        dtype_f_rest = [(f"f_rest_{i}", "f4") for i in range(45)]  # 45개의 SH 항목
        dtype_tail = [
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
        ]

        dtype_full = dtype_base + dtype_f_rest + dtype_tail

        elements = np.empty(means.shape[0], dtype=dtype_full)

        for i in range(45):
            elements[f"f_rest_{i}"] = features_extra[:, i]

        elements["x"], elements["y"], elements["z"] = (
            means[:, 0],
            means[:, 1],
            means[:, 2],
        )
        elements["nx"], elements["ny"], elements["nz"] = (
            normals[:, 0],
            normals[:, 1],
            normals[:, 2],
        )

        elements["f_dc_0"], elements["f_dc_1"], elements["f_dc_2"] = (
            colors[:, 0],
            colors[:, 1],
            colors[:, 2],
        )
        elements["opacity"] = opacities[:, 0]

        elements["scale_0"], elements["scale_1"], elements["scale_2"] = (
            scales[:, 0],
            scales[:, 1],
            scales[:, 2],
        )
        elements["rot_0"], elements["rot_1"], elements["rot_2"], elements["rot_3"] = (
            quats[:, 0],
            quats[:, 1],
            quats[:, 2],
            quats[:, 3],
        )

        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(self.fill_cfg.output_path)

        print(f"✨ Complete! Just throw it into the renderer!")


@hydra.main(version_base=None, config_path="config", config_name="filling_config")
def main(cfg: DictConfig):

    print("⚙️ [Hydra] Configuration file loading start!")

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    filler_config = FillingConfig(**config_dict)
    filler = InternalFilling(filler_config)
    gaussian = filler.execute()
    filler.save_gsplat_to_ply(gaussian)


if __name__ == "__main__":
    main()
