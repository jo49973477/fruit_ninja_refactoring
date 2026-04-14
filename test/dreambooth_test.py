import torch
import os
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline
from diffusers.utils import load_image

def main():
    # ==========================================
    # 1. 캡틴의 환경 세팅
    # ==========================================
    # 🌟 캡틴이 학습시킨 결과물(Full Pipeline)이 저장된 폴더 경로! (LoRA가 아님!)
    TRAINED_MODEL_PATH = "model/zxy_dreambooth_lora/checkpoint-1000/pipeline" 
    TEST_IMAGE_PATH = "/home/yeongyoo/03_Dataset/03_fruitninja_finetune/orange/horizontal/orange1.png" 
    UNIQUE_TOKEN = "zxy"
    CLASS_WORD = "orange"
    
    OUTPUT_DIR = "./dreambooth_test_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==========================================
    # 2. 🌟 캡틴의 '완전체' 커스텀 파이프라인 단 한 방에 로드!
    # ==========================================
    print(f"🚀 캡틴이 깎아낸 커스텀 모델 통째로 로딩 중... ({TRAINED_MODEL_PATH})")
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        TRAINED_MODEL_PATH, # 베이스 모델 대신 캡틴의 폴더를 직접 넣습니다!
        torch_dtype=torch.float16,
        safety_checker=None # (선택) 검열 필터 끄기
    ).to("cuda")

    pipe.set_progress_bar_config(disable=False)

    # ==========================================
    # 3. 입력 이미지 로드 (Depth 모델은 이미지 입력이 '필수'입니다!)
    # ==========================================
    print("📸 기준 뎁스(Depth) 이미지 로딩 중...")
    init_image = load_image(TEST_IMAGE_PATH).convert("RGB")

    # ==========================================
    # 4. 캡틴의 가혹한 프롬프트 테스트 셋
    # ==========================================
    prompts = [
        f"A vertical cross-section of {CLASS_WORD}",
        f"A horizontal cross-section of {CLASS_WORD}",
        f"A {CLASS_WORD}" # 대조군
    ]
    
    negative_prompt = "blurry, bad quality, deformed, background noise, floater, logo, text, font, letters"

    # ==========================================
    # 5. 테스트 이미지 생성
    # ==========================================
    print("\n🔥 본격적인 테스트 생성을 시작합니다!")
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] 생성 중: {prompt}")
        
        # 🌟 Depth 모델은 반드시 image 파라미터를 받아야 합니다!
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image, # 🌟 아까 주석 처리했던 거 해제!
            num_inference_steps=30,
            guidance_scale=7.5,
            strength=0.8 # 1.0에 가까울수록 원본 외곽선만 남고 안은 새로 칠함
        ).images[0]
        
        save_path = os.path.join(OUTPUT_DIR, f"test_{i}_prompt.png")
        result.save(save_path)
        print(f" 💾 저장 완료: {save_path}")

    print("\n🎉 모든 테스트가 완료되었습니다! 폴더를 확인해 보세요!")

if __name__ == "__main__":
    main()