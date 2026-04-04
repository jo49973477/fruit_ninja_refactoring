from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, model_validator
import hydra
from omegaconf import DictConfig, OmegaConf

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
    model_path: str
    output_path: str
    physics_config: str
    guidance_config: str
    white_bg: bool
    gaussian_path: str
    gaussian_orig: str

    # --- 2. 훈련 루프 제어 변수 (길바닥에 있던 애들) ---
    view_count: int = 180
    epochs: int = 400
    steps_per_c: int = 3
    spatial_lr_scale: float = 0.1

    # --- 3. 가우시안 최적화 파라미터 (호러 무비 주인공들) ---
    # Learning Rates
    position_lr_init: float = 0.001
    position_lr_final: float = 0.0002
    position_lr_delay_mult: float = 0.02
    position_lr_max_steps: int = 600
    feature_lr: float = 0.001
    opacity_lr: float = 0.01
    scaling_lr: float = 0.001
    rotation_lr: float = 0.01

    # Densification & Control
    percent_dense: float = 0.01
    density_start_iter: int = 0
    density_end_iter: int = 3000
    densification_interval: int = 50
    opacity_reset_interval: int = 700
    densify_grad_threshold: float = 0.01
    
    # other stuffs
    max_values: int = 10000
    sd_model_vertical: str = "sd2-community/stable-diffusion-2-depth"
    sd_model_horizontal: str = "sd2-community/stable-diffusion-2-depth"
    image_size: int = 512
    

class FillingConfig(BaseModel):
    model_path: str
    output_path: str
    white_br: bool = False