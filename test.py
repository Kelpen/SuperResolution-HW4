from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as trans
from DatasetLoader import TestImages
import torch
import cv2


if __name__ == '__main__':
    tt = trans.Compose([
        trans.ToTensor(),
    ])
    d = TestImages(tt)
    dl = DataLoader(d)

    # model = SimpleCNN().cuda()
    model = torch.load('model_00001000.pth').eval().cuda()

    for data in dl:
        ls, name, _ = data
        with torch.no_grad():
            pred = model(ls.cuda())
        print(name)
        cv2.imwrite(name[0], (pred.detach().cpu()[0].clip(0, 1) * 255).type(torch.uint8).numpy()[::-1].transpose((1, 2, 0)))
        # cv2.waitKey(0)
