# coding=utf-8
"""실행 환경 점검 스크립트.

패키지 버전, GPU 사용 가능 여부, 데이터셋(Dataset/PACS) 과 이미지 리스트 파일의 존재,
도메인별 이미지 수를 출력한다. 문제가 있으면 종료 코드 1 을 돌려준다.

    python check_setup.py            # PACS 점검
    python check_setup.py --dataset office-home
"""
import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DATASETS = {
    "PACS": {
        "dir": "PACS",
        "domains": ["art_painting", "cartoon", "photo", "sketch"],
        "image_lists": [f"image_list/PACS/{d}_{s}_kfold.txt"
                        for d in ["art_painting", "cartoon", "photo", "sketch"]
                        for s in ["train", "crossval", "test"]],
    },
    "office-home": {
        "dir": "OfficeHome",
        "domains": ["Art", "Clipart", "Product", "Real_World"],
        "image_lists": [f"image_list/OfficeHome/{d}{s}.txt"
                        for d in ["Art", "Clipart", "Product", "Real_World"]
                        for s in ["", "_train", "_val"]],
    },
}

PACKAGES = ["torch", "torchvision", "numpy", "sklearn", "matplotlib", "PIL", "timm", "clip"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def check_packages() -> bool:
    print("== packages ==")
    ok = True
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "(no version attr)")
            print(f"  [ok]      {name:12s} {version}")
        except ImportError:
            print(f"  [MISSING] {name:12s} -> pip install -r requirements.txt")
            ok = False
    return ok


def check_gpu() -> bool:
    print("== gpu ==")
    try:
        import torch
    except ImportError:
        print("  torch 가 없어 GPU 를 확인할 수 없습니다.")
        return False
    if not torch.cuda.is_available():
        print("  [WARN] CUDA 를 사용할 수 없습니다. 학습 스크립트는 GPU 를 전제로 합니다.")
        return False
    for i in range(torch.cuda.device_count()):
        print(f"  [ok] cuda:{i} {torch.cuda.get_device_name(i)}")
    return True


def count_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def check_dataset(name: str) -> bool:
    spec = DATASETS[name]
    ok = True
    print(f"== dataset: {name} ==")
    root = REPO_ROOT / "Dataset" / spec["dir"]
    if not root.is_dir():
        print(f"  [MISSING] {root}  (README 의 Data Preparation 참고)")
        ok = False
    else:
        for domain in spec["domains"]:
            folder = root / domain
            if folder.is_dir():
                print(f"  [ok] {domain:14s} {count_images(folder):6d} images")
            else:
                print(f"  [MISSING] {folder}")
                ok = False

    print("== image lists ==")
    missing = [p for p in spec["image_lists"] if not (REPO_ROOT / p).is_file()]
    if missing:
        ok = False
        for p in missing:
            print(f"  [MISSING] {p}")
    else:
        print(f"  [ok] {len(spec['image_lists'])} files")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="PACS", choices=list(DATASETS))
    args = parser.parse_args()

    print(f"repo root: {REPO_ROOT}")
    print(f"python   : {sys.version.split()[0]}")
    results = [check_packages(), check_gpu(), check_dataset(args.dataset)]
    if all(results):
        print("\nALL OK")
        return 0
    print("\nSOME CHECKS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
