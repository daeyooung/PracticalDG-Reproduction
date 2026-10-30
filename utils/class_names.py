# coding=utf-8
"""데이터셋별 known 클래스 이름.

CLIP 텍스트 프롬프트("a photo of a {class}")를 만들 때 사용한다.
리스트 순서는 이미지 리스트 파일의 라벨 인덱스 순서와 반드시 같아야 한다.

원본 코드는 OfficeHome 클래스 이름 54개가 CLIPBase.py / SCIPD.py 안에 각각 세 번씩
하드코딩되어 있어 PACS 등 다른 데이터셋으로 실험할 수 없었다.
이 모듈에서 데이터셋 이름으로 조회하도록 분리했다.
"""
import torch

# PACS: 라벨 0~5 known, 라벨 6 'person' 은 target 도메인에만 있는 unknown 클래스
PACS_KNOWN = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house']

# OfficeHome: 라벨 0~53 known, 54~64 unknown
OFFICEHOME_KNOWN = [
    'Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
    'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
    'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
    'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
    'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
    'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles',
]

KNOWN_CLASSNAMES = {
    "PACS": PACS_KNOWN,
    "office-home": OFFICEHOME_KNOWN,
    "mini-office-home": OFFICEHOME_KNOWN[:6],
}


def get_known_classnames(dataset: str, num_classes: int = None) -> list:
    """데이터셋의 known 클래스 이름 목록을 돌려준다. num_classes 가 주어지면 개수가 맞는지 검사한다."""
    if dataset not in KNOWN_CLASSNAMES:
        raise KeyError(
            f"'{dataset}' 의 클래스 이름이 정의되어 있지 않습니다. "
            f"utils/class_names.py 의 KNOWN_CLASSNAMES 에 추가하세요. (지원: {list(KNOWN_CLASSNAMES)})"
        )
    names = KNOWN_CLASSNAMES[dataset]
    if num_classes is not None and len(names) != num_classes:
        raise ValueError(
            f"'{dataset}' known 클래스 수 불일치: 클래스 이름 {len(names)}개, 데이터 로더 기준 {num_classes}개"
        )
    return names


def clip_text_inputs(dataset: str, device, num_classes: int = None, template: str = "a photo of a {}"):
    """known 클래스 이름을 CLIP 토큰으로 변환한 텐서 (num_classes, 77) 를 돌려준다."""
    import clip  # openai/CLIP

    names = get_known_classnames(dataset, num_classes)
    return torch.cat([clip.tokenize(template.format(c)) for c in names]).to(device)
