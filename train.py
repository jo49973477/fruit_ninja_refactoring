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

        self.gaussians_save = self.load_ply_to_gsplat(
            self.trainer_cfg.gaussian_path, self.device
        )
        self.gaussians_orig = self.load_ply_to_gsplat(
            self.trainer_cfg.gaussian_orig, self.device
        )

        for k in self.gaussians_save:
            self.gaussians_save[k].requires_grad_(True)

        self.transform = T.ToTensor()

        dev_vertical, dev_horizontal = self.devices[-2], self.devices[-1]

        try:
            print(f"🚀 Vertical SD model is assigned in {dev_vertical}...")

            # loading stable diffusion model
            self.pipe_vertical = StableDiffusionDepth2ImgPipeline.from_pretrained(
                self.trainer_cfg.sd_model_vertical
            ).to(dev_vertical)

            # setting progress bar
            self.pipe_vertical.set_progress_bar_config(disable=True)
        except Exception as e:
            raise RuntimeError(
                f"\n🚨 Failed to load vertical {self.trainer_cfg.sd_model_vertical}!\n"
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
                f"🚨 Failed to load horizontal: {self.trainer_cfg.sd_model_horizontal}"
                f"🔥 The real cause: {e}"
            ) from e

        self.strategy = DefaultStrategy(
            refine_start_iter=100, refine_stop_iter=1500, verbose=True
        )

        self.strategy_state = self.strategy.initialize_state()

        self.optimizer = {
            k: optim.Adam([v], lr=self.trainer_cfg.lrs[k], eps=1e-15)
            for k, v in self.gaussians_save.items()
        }

        self.global_step = 0

        self.center_pos = torch.tensor(self.trainer_cfg.center_pos, device=self.device)

    def load_ply_to_gsplat(
        self, ply_path: str, device="cuda"
    ) -> dict[str, nn.Parameter]:
        """loading Gaussian Splatting styled .ply file."""

        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"🚨The file {ply_path} is not found!🚨\n")

        plydata = PlyData.read(ply_path)
        v = plydata["vertex"]

        means = np.stack((v["x"], v["y"], v["z"]), axis=-1)
        scales = np.stack((v["scale_0"], v["scale_1"], v["scale_2"]), axis=-1)
        quats = np.stack((v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]), axis=-1)
        opacities = v["opacity"][..., np.newaxis]
        colors = np.stack((v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]), axis=-1)

        return {
            "means": torch.tensor(means, dtype=torch.float32, device=device),
            "scales": torch.tensor(scales, dtype=torch.float32, device=device),
            "quats": torch.tensor(quats, dtype=torch.float32, device=device),
            "opacities": torch.tensor(opacities, dtype=torch.float32, device=device),
            "colors": torch.tensor(colors, dtype=torch.float32, device=device),
        }

    def _get_optimal_devices(self, num_models=2):
        """
        getting the list of GPUs that user can use.
        """
        
        num_gpus = torch.cuda.device_count()

        if num_gpus == 0:
            print("⚠️ You have no GPU and it will be run with only CPU. Cheer up bro")
            return ["cpu"] * (num_models + 1)

        if num_gpus == 1:
            print("⚠️ You only have 1 GPU. All models will be squeezed into cuda:0")
            return ["cuda:0"] * (num_models + 1)

        main_device = ["cuda:0"]
        sub_gpus = list(range(1, num_gpus))
        allocated_devices = [
            f"cuda:{sub_gpus[i % len(sub_gpus)]}" for i in range(num_models)
        ]
        return main_device + allocated_devices

    def create_3d_grid(self, gaussians_dict: dict, grid_size: tuple):
        """Makes 3D grid from the gaussian dictionary."""

        xyz = gaussians_dict["means"]  # Shape (N, 3)

        # Get the min and max coordinates to define the grid boundaries
        min_coords = xyz.min(dim=0)[0]
        max_coords = xyz.max(dim=0)[0]

        # Calculate the dimensions of each grid cell
        cell_dimensions = (max_coords - min_coords) / torch.tensor(
            grid_size, device=self.device
        )

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

    def _get_opaque_atom_loss(self, mask=None):
        """
        The regularizer which forces Gaussians into Opaque Atoms
        """
        cfg = self.trainer_cfg

        if not cfg.opaque_atom:
            return 0.0

        opacities = torch.sigmoid(self.gaussians_save["opacities"]).squeeze(-1)
        scales = torch.exp(self.gaussians_save["scales"])

        if mask is not None:
            opacities = opacities[mask]
            scales = scales[mask]

        loss_opaque = torch.mean(opacities * (1.0 - opacities))

        loss_scale = torch.mean(scales)

        scale_mean = scales.mean(dim=-1, keepdim=True)
        loss_iso = torch.mean((scales - scale_mean) ** 2)

        return (
            cfg.lambda_opaque * loss_opaque
            + cfg.lambda_scale * loss_scale
            + cfg.lambda_iso * loss_iso
        )

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
                # voxel smoothing
                grid = self.create_3d_grid(
                    self.gaussians_save, grid_size=(512, 512, 512)
                )
                self.smooth_gaussians_in_grid(self.gaussians_save, grid)

    def _get_gsplat_camera_matrices(self, azimuth: float, elevation: float = 0.0):
        """From the azimuth and elevation of image, and the pre-defined radius
        It extracts the w2c matrix of camera
        azimuth, elevation is "degree"
        """

        width = self.trainer_cfg.image_size
        height = self.trainer_cfg.image_size
        radius = self.trainer_cfg.init_radius

        azimuth_rad = torch.deg2rad(torch.tensor(float(azimuth)))
        elevation_rad = torch.deg2rad(torch.tensor(float(elevation)))

        # Getting the x, y, z position of camera from r, theta, phi
        cam_x = radius * torch.cos(elevation_rad) * torch.sin(azimuth_rad)
        cam_y = radius * torch.sin(elevation_rad)
        cam_z = radius * torch.cos(elevation_rad) * torch.cos(azimuth_rad)
        cam_pos = torch.tensor([cam_x, cam_y, cam_z], device=self.device)

        # getting forward vector
        forward = (self.center_pos - cam_pos) / torch.norm(cam_pos)

        up = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        if torch.abs(forward[1]) > 0.99:
            up = torch.tensor([0.0, 0.0, -1.0], device=self.device)

        # making the right direction as crossing
        right = torch.linalg.cross(up, forward)
        right = right / torch.norm(right)

        # making the up direction as crossing
        up = torch.linalg.cross(forward, right)

        viewmat = torch.eye(4, device=self.device)
        viewmat[:3, 0] = right  # x-axis
        viewmat[:3, 1] = up  # y-axis
        viewmat[:3, 2] = forward  # z-axis
        viewmat[:3, 3] = cam_pos  # camera position

        fov_y = torch.deg2rad(torch.tensor(60.0))
        focal = (height / 2.0) / torch.tan(fov_y / 2.0)

        K = torch.tensor(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            device=self.device,
        )

        world_to_cam = torch.linalg.inv(viewmat)
        return world_to_cam, K

    def _convert_SH_to_RGB(
        self,
        shs: torch.Tensor,
    ):
        """
        Usually, the colour is saved as SH style in the 3DGS style. It must be converted into RGB style
        - shs : 3DGS styled color
        """

        SH_C0 = 0.28209479177387814
        colors = shs * SH_C0 + 0.5

        return torch.clamp(colors, min=0.0, max=1.0)

    def _generate_sds_reference(
        self, render_image: torch.Tensor, depth_map: torch.Tensor, step: int, mode: str
    ) -> Image.Image:
        """
        Returning the SDS loss and doing some optimization
        - render_image : rendered image from 3DGS rasterization
        - depth_map: the depth map of rendered image
        - step: current step
        - mode: "vertical" or "horizontal"
        """
        
        
        # 1. Preparing the setting
        pipe_prompt_nega = {
            "vertical": (
                self.pipe_vertical,
                self.trainer_cfg.vertical_prompt,
                self.trainer_cfg.vertical_negative_prompt,
            ),
            "horizontal": (
                self.pipe_horizontal,
                self.trainer_cfg.horizontal_prompt,
                self.trainer_cfg.horizontal_negative_prompt,
            ),
        }
        pipe, prompt, negative_prompt = pipe_prompt_nega[mode]
        device = pipe.device

        # 2. Extracting text embedding (🌟 원본 encode_prompt 활용하여 안전하게 순서 보장!)
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt, device, 1, True, negative_prompt
        )
        # CFG를 위해 Negative(Uncond) -> Positive(Text) 순서로 결합
        text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])

        target_dtype = pipe.unet.dtype

        # 3. Normalising the image tensor appropriate for SD model
        image_tensor = render_image.detach().permute(2, 0, 1).unsqueeze(0)
        image_tensor = (image_tensor * 2.0) - 1.0  # [0,1] -> [-1,1] 기적의 마법!
        image_tensor = image_tensor.to(device=device, dtype=target_dtype)

        depth_map = depth_map.to(device=device, dtype=target_dtype)

        # getting initializing latent vector
        with torch.no_grad():
            init_latents = pipe.vae.encode(image_tensor).latent_dist.sample()
            init_latents = init_latents * pipe.vae.config.scaling_factor

        # latent vector of rendered image
        init_latents = init_latents.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([init_latents], lr=self.trainer_cfg.lrs["sds"])

        min_t = int(pipe.scheduler.config.num_train_timesteps * 0.02)
        max_t = int(pipe.scheduler.config.num_train_timesteps * 0.98)

        # starting SDS step
        for _ in range(self.trainer_cfg.sds_steps):
            optimizer.zero_grad()

            # getting t value
            t = torch.randint(min_t, max_t + 1, (1,), device=device).long()

            # adding a noise into input latent vector
            noise = torch.randn_like(init_latents)
            latents_noisy = pipe.scheduler.add_noise(init_latents, noise, t)

            # preparing depth map
            depth_mask = pipe.prepare_depth_map(
                image_tensor,
                depth_map,
                1,
                True,  # do_classifier_free_guidance
                init_latents.dtype,
                device,
            )

            # 🌟 원본과 동일하게 모델 입력값 결합 및 스케일링 추가!
            latent_model_input = torch.cat([latents_noisy] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = torch.cat([latent_model_input, depth_mask], dim=1)

            # Predicting noise using UNet
            with torch.no_grad():
                noise_pred_all = pipe.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample

            # 🌟 Negative 먼저 분리!
            noise_pred_uncond, noise_pred_text = noise_pred_all.chunk(2)

            # GUIDANCE_SCALE
            noise_pred = noise_pred_uncond + self.trainer_cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # 🌟 원본의 핵심: 가중치(w) 적용 및 NaN 방지!
            alphas = pipe.scheduler.alphas_cumprod.to(device)
            w = (1 - alphas[t]).view(1, 1, 1, 1)
            grad = w * (noise_pred - noise)
            grad = torch.nan_to_num(grad)
            grad.clamp_(-1.0, 1.0)

            # Updating the latent
            target = (init_latents - grad).detach()
            loss = 0.5 * F.mse_loss(init_latents.float(), target, reduction="sum")
            loss.backward()
            optimizer.step()

        # 5. Getting final output
        with torch.no_grad():
            init_latents = init_latents / pipe.vae.config.scaling_factor
            output_image_tensor = pipe.vae.decode(init_latents).sample
            output_image = pipe.image_processor.postprocess(output_image_tensor)[0]

        return output_image

    def _get_vertical_slice_mask(
        self, pos: torch.Tensor, viewmat: torch.Tensor, slice_thickness: float = 0.01
    ):
        """
        getting mask that masks only the vectical slice of object
        - pos: positions of the 3DGS dictionary
        - view_matrix: w2v view matrix
        """

        c2w = torch.linalg.inv(viewmat)
        forward_dir = c2w[:3, 2].to(self.device)
        vec_to_pos = pos - self.center_pos
        depth_diff = torch.matmul(vec_to_pos, forward_dir)
        mask = depth_diff >= -slice_thickness

        return mask

    def _get_horizontal_slice_mask(
        self,
        pos: torch.Tensor,
        viewmat: torch.Tensor,
        slice_thickness: float = 0.01,
        custom_center: torch.Tensor = None,
    ):
        """
        getting mask that masks only the horizontal slice of object, and the viewmat
        - pos: positions of the 3DGS dictionary
        - view_matrix: w2v view matrix
        """

        c2w = torch.linalg.inv(viewmat)
        forward_dir = c2w[:3, 2].to(self.device)

        # 🌟 1. 캡틴이 이미 구해둔 '진짜 무게중심(self.center_pos)'을 기준으로 셋팅!
        if custom_center is None:
            center_pos_3d = self.center_pos.to(self.device)
        else:
            # 🌟 2. 캡틴이 "Z축(높이)만 이 위치로 잘라!" 하고 스칼라 값만 넘겼다면?
            if isinstance(custom_center, torch.Tensor) and custom_center.dim() == 0:
                # X, Y는 진짜 물체의 중심을 유지하고, Z만 캡틴이 넘겨준 값으로 갈아끼운다!
                center_pos_3d = torch.tensor(
                    [self.center_pos[0], self.center_pos[1], custom_center.item()],
                    device=self.device,
                )
            else:
                # 캡틴이 [X, Y, Z] 좌표를 통째로 넘겼을 때는 그대로 쓴다!
                center_pos_3d = custom_center

        vec_to_pos = pos - center_pos_3d
        depth_diff = torch.matmul(vec_to_pos, forward_dir)
        mask = depth_diff >= -(slice_thickness / 2.0)

        return mask

    def get_loss(
        self, img1: torch.Tensor, img2: torch.Tensor, ssim_weight: float = 0.7
    ) -> torch.Tensor:
        """
        getting the sum of SSIM loss and MSE loss
        - img1: the first image I got
        - img2: the second image I got
        """

        if img1.dim() == 3 and img1.shape[-1] == 3:
            img1 = img1.permute(2, 0, 1).unsqueeze(0)
        elif img1.dim() == 3:  # 이미 [3, H, W]라면
            img1 = img1.unsqueeze(0)

        if img2.dim() == 3 and img2.shape[0] == 3:
            img2 = img2.unsqueeze(0)
        elif img2.dim() == 3 and img2.shape[-1] == 3:  # 혹시 [H, W, 3] 이라면
            img2 = img2.permute(2, 0, 1).unsqueeze(0)

        img1 = img1.clamp(0.0, 1.0)
        img2 = img2.clamp(0.0, 1.0)

        ssim_loss = 1.0 - ssim(img1, img2, data_range=1.0, size_average=True)

        mse_loss = F.mse_loss(img1, img2)

        total_loss = (ssim_weight * ssim_loss) + ((1.0 - ssim_weight) * mse_loss)

        return total_loss

    def _train_vertical_views(self, epoch: int):
        """get all the vertical values of slices"""

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
                means=means_cs,
                quats=quats_activated,
                scales=scales_activated,
                opacities=opacity_activated,
                colors=colors_precomp_cs,
                viewmats=viewmat.unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=self.trainer_cfg.image_size,
                height=self.trainer_cfg.image_size,
                packed=True,
                backgrounds=self.background,
            )
            # =========================================================

            # 4. SD 모델 뎁스 추출
            render_image = render_image.squeeze(0)
            depth_map = F.interpolate(
                render_image.permute(2, 0, 1).unsqueeze(0),
                size=(384, 384),
                mode="bilinear",
            )[0]
            depth_map = depth_map.unsqueeze(0).to(self.devices[-1])

            self._save_render_image(render_image, f"v{i}_render")
            target_dtype = self.pipe_vertical.depth_estimator.dtype

            with torch.no_grad():
                predicted_depth = self.pipe_vertical.depth_estimator(
                    depth_map.to(dtype=target_dtype)
                ).predicted_depth
                depth_resized = F.interpolate(
                    predicted_depth.unsqueeze(0), size=(512, 512), mode="bilinear"
                ).squeeze(0)

            # 5. 레퍼런스 이미지 생성 및 타겟 세팅
            if epoch % self.trainer_cfg.sds_per_epoch == 0:
                ref_img = self._generate_sds_reference(
                    render_image,
                    depth_map=depth_resized,
                    step=self.trainer_cfg.sds_per_epoch - epoch // 100,
                    mode="vertical",
                )
                self._save_ref_image(ref_img, f"v{i}_ref")
            else:
                ref_img = self._load_cached_reference(f"v{i}_ref.png")

            ground_truth = self.transform(ref_img).to(self.device)
            loss = self.get_loss(
                render_image, ground_truth
            ) + self._get_opaque_atom_loss(mask)

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
                optimizers=self.optimizer,
                state=self.strategy_state,
                step=self.global_step,
                info=render_info,
                packed=True,
            )

            for opt in self.optimizer.values():
                opt.step()
                opt.zero_grad()

            self.global_step += 1

            pbar.set_description(
                f"🎥 [VERTICAL CROSS-SECTION] {i}/30 views training! 📉LOSS: {loss.item():.5f}"
            )

            torch.cuda.empty_cache()

    def _get_horizontal_slice_centers(self, steps: int = 70):
        """get all the horizontal values of slices"""

        # 1. 전체 가우시안 점들의 3D 위치 가져오기
        pos = self.gaussians_save["means"]

        z_coords = pos[:, 2]

        z_min = z_coords.min().item()
        z_max = z_coords.max().item()
        slice_thickness = (z_max - z_min) / steps
        slice_centers = torch.linspace(z_min, z_max, steps).to(self.device)

        return slice_centers, slice_thickness

    def _train_horizontal_views(self, epoch: int):

        viewmat, K = self._get_gsplat_camera_matrices(azimuth=0.0, elevation=90.0)
        slice_centers, slice_thickness = self._get_horizontal_slice_centers(steps=70)

        pbar = tqdm(slice_centers[10:60])

        for i, center_pos in enumerate(pbar):
            pos = self.gaussians_save["means"]
            mask_suf = self._get_horizontal_slice_mask(
                pos, viewmat, center_pos, slice_thickness
            )

            if mask_suf.sum() == 0:
                print("No Gaussian! NO GAUSSIAN!!")
                continue

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
                means=means_cs,
                quats=quats_activated,
                scales=scales_activated,
                opacities=opacity_activated,
                colors=colors_precomp_cs,
                viewmats=viewmat.unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=self.trainer_cfg.image_size,
                height=self.trainer_cfg.image_size,
                packed=True,
                backgrounds=self.background,
            )

            # 4. SD 모델 추론 (두 번째 GPU 사용)
            render_image = render_image.squeeze(0)
            depth_map = F.interpolate(
                render_image.permute(2, 0, 1).unsqueeze(0),
                size=(384, 384),
                mode="bilinear",
            )[0]
            depth_map = depth_map.unsqueeze(0).to(self.devices[-2])
            target_dtype = self.pipe_vertical.depth_estimator.dtype

            self._save_render_image(render_image, f"h{i}_render")

            with torch.no_grad():
                predicted_depth = self.pipe_horizontal.depth_estimator(
                    depth_map.to(target_dtype)
                ).predicted_depth
                depth_resized = F.interpolate(
                    predicted_depth.unsqueeze(0), size=(512, 512), mode="bilinear"
                ).squeeze(0)

            if epoch % self.trainer_cfg.sds_per_epoch == 0:
                ref_img = self._generate_sds_reference(
                    render_image,
                    depth_map=depth_resized,
                    step=self.trainer_cfg.sds_per_epoch - epoch // 100,
                    mode="horizontal",
                )
                self._save_ref_image(ref_img, f"h{i}_ref")
            else:
                ref_img = self._load_cached_reference(f"h{i}_ref.png")

            ground_truth = self.transform(ref_img).to(self.device)
            loss = self.get_loss(
                render_image, ground_truth
            ) + self._get_opaque_atom_loss(mask_suf)

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
                packed=True,
            )

            for opt in self.optimizer.values():
                opt.step()
                opt.zero_grad()

            self.global_step += 1

            pbar.set_description(
                f"🔪 [HORIZONTAL CROSS-SECTION] {i}/50 slices training! 📉LOSS: {loss.item():.5f}"
            )
            torch.cuda.empty_cache()

    def _regularize_with_original(self):
        """regularization and texture refinement"""

        pbar = tqdm(range(30))

        for i in range(30):
            # 1. 무작위 카메라 앵글 생성 (원본 코드와 동일하게 방방곡곡에서 찍음!)
            rand_azimuth = random.uniform(0, 360)
            rand_elevation = random.uniform(-90, 90)
            viewmat, K = self._get_gsplat_camera_matrices(
                azimuth=rand_azimuth, elevation=rand_elevation
            )

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
                    packed=True,
                    backgrounds=self.background,
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
                packed=True,
                backgrounds=self.background,
            )

            self._save_render_image(render_current, f"orig_render")

            loss = self.get_loss(render_current, ground_truth, ssim_weight=0.6)

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
                packed=True,
            )

            for opt in self.optimizer.values():
                opt.step()
                opt.zero_grad()

            self.global_step += 1
            torch.cuda.empty_cache()

            pbar.set_description(
                f"🛡️ [TEXTURE REFINEMENT] {i}/30 For spatial consistency! 📉LOSS: {loss.item():.5f}"
            )


    def _load_cached_reference(self, filename: str):

        output_dir = (
            self.trainer_cfg.output_path
            if hasattr(self.trainer_cfg, "output_path")
            else "./output"
        )
        filepath = os.path.join(output_dir, "output_ref", filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"🚨 There is no Cached Reference! {filepath}")

        return Image.open(filepath)


    def _save_render_image(self, render_image, view_name):

        output_dir = (
            self.trainer_cfg.output_path
            if hasattr(self.trainer_cfg, "output_path")
            else "./output"
        )
        os.makedirs(os.path.join(output_dir, "output_renders"), exist_ok=True)

        if render_image.dim() == 4:
            render_image = render_image.squeeze(0)

        img_to_save = render_image.detach().cpu().permute(2, 0, 1)

        torchvision.utils.save_image(
            img_to_save, os.path.join(output_dir, "output_renders", f"{view_name}.png")
        )


    def _save_ref_image(self, ref_image, view_name):

        output_dir = (
            self.trainer_cfg.output_path
            if hasattr(self.trainer_cfg, "output_path")
            else "./output"
        )
        os.makedirs(os.path.join(output_dir, "output_ref"), exist_ok=True)

        ref_image.save(os.path.join(output_dir, "output_ref", f"{view_name}.png"))


    def _save_gaussian_ply(self, epoch):

        output_dir = (
            self.trainer_cfg.output_path
            if hasattr(self.trainer_cfg, "output_path")
            else "./output"
        )
        os.makedirs(os.path.join(output_dir, "output_models"), exist_ok=True)

        save_dict = {k: v.detach().cpu() for k, v in self.gaussians_save.items()}
        torch.save(
            save_dict,
            os.path.join(output_dir, "output_models", f"orange_ninja_epoch_{epoch}.pt"),
        )

        print(
            f"💾 [저장 완료] Epoch {epoch} - 가우시안 모델이 안전하게 보관되었습니다!"
        )


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

    print("⚙️ [Hydra] Completed File Loading!")

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    trainer_config = TrainerConfig(**config_dict)
    trainer = Trainer(trainer_config)
    trainer.train()


if __name__ == "__main__":
    main()
