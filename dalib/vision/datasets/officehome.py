import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class OfficeHome(ImageList):

    CLASSES = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
               'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
               'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
               'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
               'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
               'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
               'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker']

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

        assert task in self.image_list
        data_list_file = os.path.join('/home/user/Desktop/dae/HDG-main', self.image_list[task])

        super(OfficeHome, self).__init__(root, num_classes=len(filter_class), data_list_file=data_list_file,
                                       filter_class=filter_class, **kwargs)

        self.domain = ["Art", "Clipart", "Product", "Real_World"]

    def __getitem__(self, index):
        """
        Parameters:
            - **index** (int): Index
            - **return** (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.data_list[index]
        path = path.replace('OfficeHom', 'OfficeHome')
        domain_name = path.split('/')[3]
        domain_label = self.domain.index(domain_name)
        img = self.loader(os.path.join('/home/user/Desktop/dae/HDG-main',path[1:]))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target, domain_label