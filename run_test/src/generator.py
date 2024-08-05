import cv2
import os


class DictGenerator():
    def __init__(self, sample_data: dict, params: dict = {}) -> None:
        self.sample_data = sample_data
        self.params = params


    def __len__(self):
        raise NotImplementedError


    def __call__(self):
        raise NotImplementedError


class ImageGenerator(DictGenerator):
    def __len__(self):
        return len(self.sample_data)


    def __call__(self):
        for fname in self.sample_data:
            lr_img = cv2.imread(os.path.join(self.params['lr_dir'], fname))
            hr_img = cv2.imread(os.path.join(self.params['hr_dir'], fname))
            yield fname, lr_img, hr_img
