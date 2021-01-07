from torch.utils.data.dataloader import Dataset, DataLoader
from torchvision import transforms as trans
from PIL import Image
import torch
import cv2
import torch.nn.functional as F


class TrainingImages(Dataset):
    def __init__(self, transform):
        with open('training.txt') as f:
            self.data_list = f.readlines()
        self.data_num = len(self.data_list)
        self.transform = transform
        self.scale = 3

    def __getitem__(self, item):
        filename = 'training_hr_images/' + self.data_list[item][:-1]
        img_hr = Image.open(filename)
        img_hr = self.transform(img_hr)
        h, w = list(img_hr.shape[-2:])
        img_size_lr = (h//3, w//3)
        img_size_hr = (img_size_lr[0]*3, img_size_lr[1]*3)
        img_lr = F.interpolate(img_hr[None], size=img_size_lr, mode='bicubic')[0]
        img_hr = F.interpolate(img_hr[None], size=img_size_hr)[0]
        return img_lr, img_hr, self.data_list[item][:-1], self.scale

    def __len__(self):
        return self.data_num


class TestImages(Dataset):
    def __init__(self, transform):
        with open('testing.txt') as f:
            self.data_list = f.readlines()
        self.data_num = len(self.data_list)
        self.transform = transform
        self.scale = 3

    def __getitem__(self, item):
        filename = 'testing_lr_images/' + self.data_list[item][:-1]
        img_hr = Image.open(filename)
        img_lr = self.transform(img_hr)
        return img_lr, self.data_list[item][:-1], self.scale

    def __len__(self):
        return self.data_num


if __name__ == '__main__':
    tt = trans.Compose([
        trans.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01),
        trans.RandomHorizontalFlip(),
        trans.RandomVerticalFlip(),
        trans.ToTensor(),
    ])
    d = TrainingImages(tt)
    dl = DataLoader(d)
    a = iter(dl)
    for dd in a:
        _, d, _, _ = dd
        cv2.imshow('a', (d[0]*255).type(torch.uint8).numpy()[::-1].transpose((1, 2, 0)))
        cv2.waitKey(0)
