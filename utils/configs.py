from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, model_validator
import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field

# --- Sub-models ---

class AdditionalMaterialParam(BaseModel):
    point: List[float]
    size: List[float]
    E: float
    nu: float
    density: Optional[float] = None # 나중에 부모 density로 덮어씌워질 예정



class ParticleFillingParams(BaseModel):
    n_grid: Optional[int] = None # 나중에 부모 n_grid * 4 로 설정됨
    density_threshold: float = 5.0
    search_threshold: float = 3.0
    max_particles_num: int = 2000000
    max_partciels_per_cell: int = 1  # (원본 오타 유지: partciels -> particles로 고치고 싶지만 일단 원본 존중)
    search_exclude_direction: int = 5
    ray_cast_direction: int = 4
    boundary: Optional[List[float]] = None
    smooth: bool = False
    visualize: bool = False

# --- Main Parameter Models ---



class MaterialParams(BaseModel):
    material: str = "jelly"
    grid_lim: float = 2.0
    n_grid: int = 50
    nu: float = Field(0.4, le=0.5, ge=0.0) # Poisson's ratio 제한
    E: float = 1e5
    g: float = 9.8
    density: float = 200.0
    
    # Optional 파라미터들 (원본 코드에 기본값이 없고 if만 있는 애들)
    yield_stress: Optional[float] = None
    hardening: Optional[float] = None
    xi: Optional[float] = None
    friction_angle: Optional[float] = None
    plastic_viscosity: Optional[float] = None
    rpic_damping: Optional[float] = None
    pic_damping: Optional[float] = None
    softening: Optional[float] = None
    opacity_threshold: Optional[float] = None
    grid_v_damping_scale: Optional[float] = None
    
    additional_material_params: Optional[List[AdditionalMaterialParam]] = None



class CameraParams(BaseModel):
    mpm_space_viewpoint_center: List[float] = [1.0, 1.0, 1.0]
    mpm_space_vertical_upward_axis: List[float] = [0, 0, 1]
    default_camera_index: int = 0
    show_hint: bool = False
    init_azimuthm: Optional[float] = None
    init_elevation: Optional[float] = None
    init_radius: Optional[float] = None
    delta_a: Optional[float] = None
    delta_e: Optional[float] = None
    delta_r: Optional[float] = None
    move_camera: bool = False



class PreprocessingParams(BaseModel):
    opacity_threshold: float = 0.02
    rotation_degree: List[float] = []
    rotation_axis: List[float] = []
    sim_area: Optional[List[float]] = None
    particle_filling: Optional[ParticleFillingParams] = None



class TimeParams(BaseModel):
    substep_dt: float = 1e-4
    frame_dt: float = 1e-2
    frame_num: int = 100



class BCParams(BaseModel):
    # Boundary Conditions는 리스트 형태의 딕셔너리로 들어오는 구조임 (원본 확인 완)
    conditions: List[Dict[str, Any]] = []

# --- Root Config Model ---



class PhysicsConfig(BaseModel):
    material_params: MaterialParams = MaterialParams()
    bc_params: BCParams = BCParams()
    time_params: TimeParams = TimeParams()
    preprocessing_params: PreprocessingParams = PreprocessingParams()
    camera_params: CameraParams = CameraParams()
    
    @model_validator(mode='after')
    def set_dependent_defaults(self):
        # 캡틴의 완벽한 로직 그대로!
        if self.preprocessing_params.particle_filling:
            if self.preprocessing_params.particle_filling.n_grid is None:
                self.preprocessing_params.particle_filling.n_grid = self.material_params.n_grid * 4
        
        # 반드시 자기 자신(self)을 리턴해줘야 완성이 됨!
        return self
    


class TrainerConfig(BaseModel):
    # --- 1. 기본 경로 및 파일 설정 (원래 있던 거) ---
    output_path: str
    white_bg: bool
    gaussian_path: str
    gaussian_orig: str
    lora_path: Optional[str] = None
    
    epochs: int = 400
    init_radius: float = 2.5
    image_size: int = 512
    sds_per_epoch: int = 10
    sds_steps: int = 5
    guidance_scale: int = 15
    
    lrs: Dict[str, float] = field(default_factory=lambda: {
        "means": 1.6e-4,    
        "scales": 1e-3,    
        "quats": 1e-3,      
        "opacities": 1e-2,  
        "colors": 2.5e-3
    })
    
    sd_model_vertical: str = "sd2-community/stable-diffusion-2-depth"
    sd_model_horizontal: str = "sd2-community/stable-diffusion-2-depth"
    
    vertical_prompt: str=  "A high-quality, ultra-realistic photo of an orange, detailed orange peel texture, perfect lighting, 8k resolution"
    vertical_negative_prompt: str = "low resolution, blurry, distorted, cross-section, cut, flesh, inside"
    horizontal_prompt: str = "A high-quality, ultra-realistic macro photo of an orange cross-section, juicy orange flesh, distinct citrus segments, 8k resolution"
    horizontal_negative_prompt: str = "low resolution, blurry, peel only, whole orange, distorted, fake"
    


class OldFinetuneConfig(BaseModel):
    nickname: Optional[str] = None
    class_prompt: str
    save_dir: str = "./model"
    load_dir: Optional[str] = None
    image_dir: str
    output_image_save_dir: str
    sd_model: str = "sd2-community/stable-diffusion-2-depth"
    prior_loss_weight: float = 1.0
    num_train_epochs: int = 100
    learning_rate: float = 5e-6
    train_batch_size: int = 4


class FinetuneConfig(BaseModel):
    # ==========================================
    # 1. 모델 경로 세팅 (Model Settings)
    # ==========================================
    pretrained_model_name_or_path: str = "sd2-community/stable-diffusion-2-base"
    pretrained_txt2img_model_name_or_path: str = "sd2-community/stable-diffusion-2-base"
    revision: Optional[str] = None
    tokenizer_name: Optional[str] = None

    # ==========================================
    # 2. 데이터 세팅 (Data Settings)
    # ==========================================
    instance_data_dir: str = "./data/zxy_images"
    instance_prompt: str = "A zxy screw"
    class_data_dir: Optional[str] = "./data/class_images"
    nickname: str
    class_prompt: Optional[str] = "A screw"
    num_class_images: int = 100

    # ==========================================
    # 3. 해상도 및 처리 (Image Processing)
    # ==========================================
    resolution: int = 512
    center_crop: bool = False

    # ==========================================
    # 4. 학습 뼈대 세팅 (Training Basics)
    # ==========================================
    output_dir: str = "./model/zxy_dreambooth_lora"
    seed: int = 42
    train_text_encoder: bool = False
    train_batch_size: int = 4
    sample_batch_size: int = 4
    num_train_epochs: int = 1
    max_train_steps: int = 1000
    checkpointing_steps: int = 500
    resume_from_checkpoint: Optional[str] = None

    # ==========================================
    # 5. 최적화 및 VRAM 다이어트 (Optimization & Memory)
    # ==========================================
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    mixed_precision: str = "fp16"
    use_8bit_adam: bool = False

    # ==========================================
    # 6. 학습률 및 옵티마이저 (Learning Rate & Optimizer)
    # ==========================================
    learning_rate: float = 5e-5
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0

    # ==========================================
    # 7. 사전 지식 보존 (Prior Preservation)
    # ==========================================
    with_prior_preservation: bool = False
    prior_loss_weight: float = 1.0

    # ==========================================
    # 8. 허깅페이스 허브 및 분산 학습 (System)
    # ==========================================
    push_to_hub: bool = False
    hub_token: Optional[str] = None
    hub_model_id: Optional[str] = None
    logging_dir: str = "logs"
    local_rank: int = -1
    