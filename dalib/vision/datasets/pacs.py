import os
from pathlib import Path

from .imagelist import ImageList

# 저장소 루트 (dalib/vision/datasets/pacs.py 기준 세 단계 위).
# 원본 코드는 특정 PC의 절대 경로('/home/user/Desktop/dae/HDG-main')가 하드코딩되어 있어
# 다른 환경에서는 실행되지 않았다. 저장소 위치를 기준으로 경로를 해석하도록 수정했다.
REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: str) -> str:
    """경로가 실제로 존재하면 그대로 쓰고, 아니면 저장소 루트 기준 상대 경로로 해석한다.

    예) '/Dataset/PACS/photo/dog/056_0022.jpg' -> '<repo>/Dataset/PACS/photo/dog/056_0022.jpg'
    """
    if os.path.exists(path):
        return path
    return str(REPO_ROOT / path.lstrip("/"))


def domain_label_from_path(path: str, domains) -> int:
    """이미지 경로에 포함된 폴더 이름으로 도메인 인덱스를 찾는다."""
    parts = path.replace("\\", "/").split("/")
    for part in parts:
        if part in domains:
            return domains.index(part)
    raise ValueError(f"도메인 이름을 경로에서 찾을 수 없습니다: {path}")


class PACS(ImageList):
    """PACS 데이터셋 (7 classes, 4 domains).

    라벨 0~5 (dog, elephant, giraffe, guitar, horse, house)는 known,
    라벨 6 (person)은 target 도메인에만 존재하는 unknown 클래스로 사용한다.
    """

    CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
    DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]

    def __init__(self, root, task, filter_class, split='all', **kwargs):
        if split == 'all':
            self.image_list = {
                "A": "image_list/PACS/art_painting_test_kfold.txt",
                "C": "image_list/PACS/cartoon_test_kfold.txt",
                "P": "image_list/PACS/photo_test_kfold.txt",
                "S": "image_list/PACS/sketch_test_kfold.txt",
            }
        elif split == 'train':
            self.image_list = {
                "A": "image_list/PACS/art_painting_train_kfold.txt",
                "C": "image_list/PACS/cartoon_train_kfold.txt",
                "P": "image_list/PACS/photo_train_kfold.txt",
                "S": "image_list/PACS/sketch_train_kfold.txt",
            }
        elif split == 'val':
            self.image_list = {
                "A": "image_list/PACS/art_painting_crossval_kfold.txt",
                "C": "image_list/PACS/cartoon_crossval_kfold.txt",
                "P": "image_list/PACS/photo_crossval_kfold.txt",
                "S": "image_list/PACS/sketch_crossval_kfold.txt",
            }
        else:
            raise ValueError(f"unknown split: {split}")

        assert task in self.image_list
        data_list_file = resolve_path(self.image_list[task])

        super(PACS, self).__init__(root, num_classes=len(filter_class), data_list_file=data_list_file,
                                   filter_class=filter_class, **kwargs)
        self.domain = self.DOMAINS

    def __getitem__(self, index):
        """
        Parameters:
            - **index** (int): Index
            - **return** (tuple): (image, target, domain_label)
        """
        path, target = self.data_list[index]
        domain_label = domain_label_from_path(path, self.domain)
        img = self.loader(resolve_path(path))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target, domain_label
