from .imagelist import ImageList
from .pacs import resolve_path, domain_label_from_path


class OfficeHome(ImageList):
    """OfficeHome 데이터셋 (65 classes, 4 domains).

    라벨 0~53은 known, 54~64는 target 도메인에만 존재하는 unknown 클래스로 사용한다.
    경로는 저장소 루트 기준으로 해석한다 (pacs.resolve_path 참고).
    """

    CLASSES = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
               'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
               'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
               'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
               'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
               'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
               'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker']
    DOMAINS = ["Art", "Clipart", "Product", "Real_World"]

    def __init__(self, root, task, filter_class, split='all', **kwargs):
        if split == 'all':
            self.image_list = {
                "A": "image_list/OfficeHome/Art.txt",
                "C": "image_list/OfficeHome/Clipart.txt",
                "P": "image_list/OfficeHome/Product.txt",
                "R": "image_list/OfficeHome/Real_World.txt",
            }
        elif split == 'train':
            self.image_list = {
                "A": "image_list/OfficeHome/Art_train.txt",
                "C": "image_list/OfficeHome/Clipart_train.txt",
                "P": "image_list/OfficeHome/Product_train.txt",
                "R": "image_list/OfficeHome/Real_World_train.txt",
            }
        elif split == 'val':
            self.image_list = {
                "A": "image_list/OfficeHome/Art_val.txt",
                "C": "image_list/OfficeHome/Clipart_val.txt",
                "P": "image_list/OfficeHome/Product_val.txt",
                "R": "image_list/OfficeHome/Real_World_val.txt",
            }
        else:
            raise ValueError(f"unknown split: {split}")

        assert task in self.image_list
        data_list_file = resolve_path(self.image_list[task])

        super(OfficeHome, self).__init__(root, num_classes=len(filter_class), data_list_file=data_list_file,
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
