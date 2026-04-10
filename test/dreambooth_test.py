import torch
import os
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline, StableDiffusionPipeline
from diffusers.utils import load_image

def main():
    # ==========================================
    # 1. 캡틴의 경로에 맞게 수정해주세요!
    # ==========================================
    BASE_MODEL = "sd2-community/stable-diffusion-2-base"
    DEPTH_MODEL=  "sd2-community/stable-diffusion-2-depth"
    LORA_PATH = "./model/zxy_train2_5e5/adapter_model.safetensors" # 🌟 학습된 LoRA 가중치 폴더 경로
    TEST_IMAGE_PATH = "/home/yeongyoo/03_Dataset/01_t-less_v2/train_kinect/01/rgb/0637.png" # 🌟 COLMAP 원본 사진 중 하나 (Depth 추출용)
    UNIQUE_TOKEN = "zxy" # 🌟 캡틴이 학습시킬 때 쓴 특수 단어
    CLASS_WORD = "screw"
    GIVE_DEPTH = False
    
    OUTPUT_DIR = "./dreambooth_test_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==========================================
    # 2. 파이프라인 및 로라(LoRA) 로드 (어제 고생한 부분 완벽 방어!)
    # ==========================================
    print("🚀 베이스 모델 로딩 중...")
    pipe_nodepth = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16
    ).to("cuda")
    
    pipe_depth =  StableDiffusionDepth2ImgPipeline.from_pretrained(
        DEPTH_MODEL, torch_dtype=torch.float16
    ).to("cuda")
    
    pipe = pipe_depth if GIVE_DEPTH else pipe_nodepth

    print(f"🔗 LoRA 가중치 로딩 중... ({LORA_PATH})")
    pipe.load_lora_weights(LORA_PATH)
    
    # 활성화된 어댑터 강제 적용 (안전 장치)
    active_adapters = pipe.get_active_adapters()
    if active_adapters:
        pipe.set_adapters(active_adapters, adapter_weights=[1.0])
        print(f"✅ LoRA 어댑터 적용 완료: {active_adapters}")
    else:
        print("⚠️ [경고] 활성화된 LoRA 어댑터가 없습니다!")

    pipe.set_progress_bar_config(disable=False)

    # ==========================================
    # 3. 입력 이미지 로드 (Depth 정보 추출의 뼈대가 됨)
    # ==========================================
    init_image = load_image(TEST_IMAGE_PATH) if GIVE_DEPTH else Image.new("RGB", (512, 512), color=(128, 128, 128))

    # ==========================================
    # 4. 캡틴의 가혹한(?) 프롬프트 테스트 셋
    # ==========================================
    # 다양한 상황을 주었을 때 '나사'의 질감을 잃어버리는지 테스트합니다.
    prompts = [
        f"A photo of {UNIQUE_TOKEN} {CLASS_WORD}", # 1. 기본 테스트 (원본 질감이 잘 나오는지)
        f"A {UNIQUE_TOKEN} {CLASS_WORD} under dramatic dark lighting", # 2. 조명 반응 테스트
        f"A {UNIQUE_TOKEN} {CLASS_WORD} completely made of pure gold", # 3. 재질 변형 테스트 (과적합 확인용)
        f"A vertical cross-section of {UNIQUE_TOKEN} {CLASS_WORD}",
        f"A horizontal cross-section of {UNIQUE_TOKEN} {CLASS_WORD}",
        f"A {CLASS_WORD}" # 4. 대조군 (특수 단어가 없을 때는 그냥 밋밋한 나사가 나와야 함)
    ]
    
    negative_prompt = "blurry, bad quality, deformed, background noise, floater, logo, text, font, letters"

    # ==========================================
    # 5. 테스트 이미지 생성
    # ==========================================
    print("\n🔥 본격적인 테스트 생성을 시작합니다!")
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] 생성 중: {prompt}")
        
        # strength: 1.0에 가까울수록 원본 이미지를 무시하고 새로 그림 (통상적으로 0.8 추천)
        result = pipe(
            prompt = prompt,
            negative_prompt=negative_prompt,
            # image = init_image,
            num_inference_steps=30,
            guidance_scale=7.5,
            strength = 0.8
        ).images[0]
        
        save_path = os.path.join(OUTPUT_DIR, f"test_{i}_prompt.png")
        result.save(save_path)
        print(f" 💾 저장 완료: {save_path}")

    print("\n🎉 모든 테스트가 완료되었습니다! 폴더를 확인해 보세요!")

if __name__ == "__main__":
    main()