import os
import gc
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from argparse import Namespace

# 🌟 아까 캡틴이 받아둔 HF 공식 스크립트에서 main 함수를 슬쩍 훔쳐옵니다!
from utils.train_dreambooth_depth import main as hf_main

@hydra.main(version_base=None, config_path="config", config_name="finetune")
def hydra_main(cfg: DictConfig):
    print("🚀 삐리빅! 로보코 센세의 '3단 콤보 Hydra 지휘소' 가동!")
    
    # 1. Hydra 설정을 조작하기 편한 파이썬 딕셔너리로 변환
    base_config = OmegaConf.to_container(cfg, resolve=True)
    
    # base_dir은 캡틴의 이미지 폴더 최상위 경로 (예: "./data/images")
    base_dir = base_config["instance_data_dir"] 
    
    nickname = base_config.get("nickname", "zxy")
    class_prompt = base_config.get("class_prompt", "screw")
    
    # 2. 캡틴의 3단계 시나리오 세팅!
    stages = [
        {
            "folder": "vanilla", 
            "prompt": f"A {nickname} {class_prompt}", 
            "desc": f"🌟Teaching {nickname}, the new {class_prompt}..."
        },
        {
            "folder": "vertical", 
            "prompt": f"A vertical cross-section of a {nickname} {class_prompt}", 
            "desc": "🎥Vertical view fine-tuning started..."
        },
        {
            "folder": "horizontal", 
            "prompt": f"A horizontal cross-section of a {nickname} {class_prompt}", 
            "desc": "🔪Horizontal view fine-tuning started..."
        }
    ]

    # 3. 3단계 지옥의 훈련 루프 시작!
    for stage in stages:
        print(f"\n{'='*50}")
        print(f"{stage['desc']}")
        print(f"🍎 Let's start training of {stage['folder']}!")
        print(f"{'='*50}")
        
        # 🌟 핵심 마법: 현재 스테이지에 맞게 경로와 프롬프트를 '동적'으로 덮어쓰기!
        current_cfg = base_config.copy()
        current_cfg["instance_data_dir"] = os.path.join(base_dir, stage["folder"])
        current_cfg["instance_prompt"] = stage["prompt"]
        
        # 덮어쓴 딕셔너리를 HF 코드가 속아 넘어갈 Namespace로 변신!
        args = Namespace(**current_cfg)
        
        # 🌟 HF 공식 코드 강제 호출! (알아서 훈련하고 가중치 저장함)
        hf_main(args)
        
        # 🚨 [초긴급 주의사항] 🚨
        # 파이썬 루프 안에서 HF 코드를 연속으로 부르면 VRAM에 쓰레기가 쌓여서 폭발(OOM)할 수 있어!
        # 한 단계 끝날 때마다 변기 물 내리듯이 VRAM을 싹 비워줘야 해!
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print("\n🎉 삐리빅! 3단계 콤보 학습이 완벽하게 종료되었습니다! 캡틴, 퇴근해(종발)!!")

if __name__ == "__main__":
    hydra_main()