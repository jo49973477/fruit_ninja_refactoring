from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, model_validator
import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field

class AdditionalMaterialParam(BaseModel):
    point: List[float]
    size: List[float]
    E: float
    nu: float
    density: Optional[float] = None # 나중에 부모 density로 덮어씌워질 예정


class ParticleFillingParams(BaseModel):
    n_grid: Optional[int] = 200
    density_threshold: float = 5.0
    search_threshold: float = 3.0
    max_particles_num: int = 19000000
    max_partciels_per_cell: int = 1  # (원본 오타 유지: partciels -> particles로 고치고 싶지만 일단 원본 존중)
    search_exclude_direction: int = 2
    ray_cast_direction: int = 3
    boundary: Optional[List[float]] = [0.51, 1.50, 0.57, 1.43, 0.51, 1.49]
    smooth: bool = False




class TrainerConfig(BaseModel):
    # --- 1. 기본 경로 및 파일 설정 (원래 있던 거) ---
    output_path: str
    white_bg: bool
    gaussian_path: str
    gaussian_orig: str
    lora_path: Optional[str] = None
    center_pos: List[float] = [0.0, 0.0, 0.0]

    epochs: int = 400
    init_radius: float = 2.5
    image_size: int = 512
    sds_per_epoch: int = 10
    sds_steps: int = 5
    guidance_scale: float = 15

    lrs: Dict[str, float] = field(
        default_factory=lambda: {
            "means": 1.6e-4,
            "scales": 1e-3,
            "quats": 1e-3,
            "opacities": 1e-2,
            "colors": 2.5e-3,
            "sds": 1e-3,
        }
    )

    sd_model_vertical: str = "sd2-community/stable-diffusion-2-depth"
    sd_model_horizontal: str = "sd2-community/stable-diffusion-2-depth"

    vertical_prompt: str = "A high-quality, ultra-realistic photo of an orange, detailed orange peel texture, perfect lighting, 8k resolution"
    vertical_negative_prompt: str = (
        "low resolution, blurry, distorted, cross-section, cut, flesh, inside"
    )
    horizontal_prompt: str = "A high-quality, ultra-realistic macro photo of an orange cross-section, juicy orange flesh, distinct citrus segments, 8k resolution"
    horizontal_negative_prompt: str = (
        "low resolution, blurry, peel only, whole orange, distorted, fake"
    )

    lambda_opaque: float = 0.1
    lambda_scale: float = 0.01
    lambda_iso: float = 0.1
    opaque_atom: bool = False


class FillingConfig(BaseModel):
    # ==========================================
    # 1. File I/O
    # ==========================================
    model_path: str = Field(
        ..., description="Path to the input PLY file (Original 3DGS model)"
    )
    output_path: str = Field(
        ..., description="Path to save the filled PLY file"
    )

    # ==========================================
    # 2. Rendering Setup 
    # ==========================================
    white_br: bool = Field(
        default=True, 
        description="Set to True for a white background, False for black (matches background tensor)"
    )
    
    rotation_degree: List[float] = [0.0]
    rotation_axis: List[int] = [0]

    # ==========================================
    # 3. Filling & Physics Parameters 
    # ==========================================
    
    particle_params: ParticleFillingParams = Field(default_factory=ParticleFillingParams)
    


class FinetuneConfig(BaseModel):
    # ==========================================
    # 1. Model Settings
    # ==========================================
    pretrained_model_name_or_path: str = "sd2-community/stable-diffusion-2-depth"
    pretrained_txt2img_model_name_or_path: str = "sd2-community/stable-diffusion-2-base"
    revision: Optional[str] = None

    # ==========================================
    # 2. Data Settings
    # ==========================================
    instance_data_dir: str = "./data/zxy_images"
    class_data_dir: Optional[str] = "./data/class_images"
    prompt_json_dir: Optional[str] = None
    nickname: Optional[str]
    class_prompt: Optional[str]
    num_class_images: int = 100

    # ==========================================
    # 3. Image Processing
    # ==========================================
    resolution: int = 512

    # ==========================================
    # 4. Training Basics
    # ==========================================
    output_dir: str = "./model/zxy_dreambooth_lora"
    seed: int = 42
    train_text_encoder: bool = False
    train_batch_size: int = 4
    num_train_epochs: int = 1
    max_train_steps: int = 1000
    checkpointing_steps: int = 500
    resume_from_checkpoint: Optional[str] = None

    # ==========================================
    # 5. Optimization & Memory
    # ==========================================
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    mixed_precision: str = "fp16"
    use_8bit_adam: bool = False

    # ==========================================
    # 6. Learning Rate & Optimizer
    # ==========================================
    learning_rate: float = 5e-5
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0

    # ==========================================
    # 7. Prior Preservation
    # ==========================================
    with_prior_preservation: bool = False
    prior_loss_weight: float = 1.0

    # ==========================================
    # 8. System
    # ==========================================
    logging_dir: str = "logs"