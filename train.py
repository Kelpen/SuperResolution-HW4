from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as trans
from DatasetLoader import TrainingImages, TestImages
from Models.SR_Model import SR_Model, Discriminator
from torch.nn import MSELoss
from torch import optim
import torch
import cv2


if __name__ == '__main__':
    show_taining = False
    
    tt = trans.Compose([
        trans.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.4),
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
    ])
    d = TrainingImages(tt)
    dl = DataLoader(d, shuffle=True, num_workers=2, batch_size=1)

    model = SR_Model().cuda()
    disc = Discriminator().cuda()
    # model = torch.load('model_00002000.pth').cuda()
    # disc = torch.load('disc_00002000.pth').cuda()
    
    loss_func = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=3.0e-4)
    optimizer_d = optim.Adam(disc.parameters(), lr=3.0e-4)

    counter = 1
    pos = torch.ones((1, 1024)).cuda()
    neg = torch.zeros((1, 1024)).cuda()
    for epoch in range(100):
        for data in dl:
            ls, hs, _, _ = data
            ls = ls.cuda()
            hs = hs.cuda()

            pred = model(ls)
            d_sr_score = disc(pred)
            loss_g = loss_func(d_sr_score, pos)
            loss = loss_func(pred, hs) * 300 + loss_g

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # discriminator
            d_hs_score = disc(hs)
            d_sr_score = disc(pred.detach())
            loss_d = loss_func(d_hs_score, pos) + loss_func(d_sr_score, neg)

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            if counter % 10 == 0:
                print(counter, 'loss pixel:', loss.detach().cpu(), 'loss disc:', loss_d.detach().cpu(), 'loss gen:', loss_g.detach().cpu())
                # print(counter, loss.detach().cpu())
            if counter % 100 == 0:
                if show_taining:
                    cv2.imshow('a', (pred.detach().cpu()[0].clip(0, 1) * 255).type(torch.uint8).numpy()[::-1].transpose((1, 2, 0)))
                    cv2.imshow('g', (hs[0].cpu() * 255).type(torch.uint8).numpy()[::-1].transpose((1, 2, 0)))
                    cv2.waitKey(10)
            if counter % 1000 == 0:
                torch.save(model, 'model_%08d.pth' % counter)
                torch.save(disc, 'disc_%08d.pth' % counter)
            counter += 1

