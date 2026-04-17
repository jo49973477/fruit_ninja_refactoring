import torch
import os
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline
from diffusers.utils import load_image


def main():
    # ==========================================
    # 1. 캡틴의 환경 세팅 (순정 베이스 모델 버전!)
    # ==========================================
    # 🌟 훈련 전 순정 상태의 공식 Depth 모델 경로를 직접 입력!
    BASE_MODEL_PATH = "sd2-community/stable-diffusion-2-depth"
    TEST_IMAGE_PATH = (
        "/home/yeongyoo/03_Dataset/01_t-less_v2/train_canon/01/rgb/0636.jpg"
    )
    CLASS_WORD = "zxy screw"
    OUTPUT_DIR = "./dreambooth_test_results_base_zxy"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==========================================
    # 2. '순정' 파이프라인 단 한 방에 로드!
    # ==========================================
    print(f"🚀 순정 베이스 모델 통째로 로딩 중... ({BASE_MODEL_PATH})")
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")

    pipe.set_progress_bar_config(disable=False)

    # ==========================================
    # 3. 입력 이미지 로드
    # ==========================================
    print("📸 기준 뎁스(Depth) 원본 이미지 로딩 중...")
    init_image = load_image(TEST_IMAGE_PATH).convert("RGB")

    # ==========================================
    # 4. 프롬프트 테스트 셋 (순정 모델이므로 고유 토큰은 쓰지 않습니다!)
    # ==========================================
    prompts = [
        f"A vertical cross-section of {CLASS_WORD}",
        f"A horizontal cross-section of {CLASS_WORD}",
        f"A {CLASS_WORD},"  # 대조군
        f"A top view of a vertical cross-section of zxy",
        f"A top view of a horizontal cross-section of zxy",
        f"A top view of a zxy,",  # 대조군
    ]

    negative_prompt = "blurry, bad quality, deformed, background noise, floater, logo, text, font, letters"

    # ==========================================
    # 5. 테스트 이미지 생성
    # ==========================================
    print("\n🔥 본격적인 순정 모델 테스트 생성을 시작합니다!")
    for i, prompt in enumerate(prompts):
        print(f"[{i + 1}/{len(prompts)}] 생성 중: {prompt}")

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            num_inference_steps=30,
            guidance_scale=4.75,
            strength=1.0,
        ).images[0]

        save_path = os.path.join(OUTPUT_DIR, f"base_test_{i}_prompt.png")
        result.save(save_path)
        print(f" 💾 저장 완료: {save_path}")

    print(
        "\n🎉 순정 모델 테스트가 완료되었습니다! 훈련된 결과와 나란히 띄워놓고 비교해 보세요!"
    )


if __name__ == "__main__":
    main()
