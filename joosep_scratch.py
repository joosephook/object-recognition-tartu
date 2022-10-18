import torch
import torchvision

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from PIL import Image
import pandas as pd
import numpy as np

import torchvision.transforms.functional as fn


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return fn.pad(image, padding, 0, 'constant')


class TartuObjectDataset(torch.utils.data.Dataset):
    def __init__(self, folder='images', transform=None):
        self.folder = folder
        self.img_ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

        df = pd.read_csv('train.csv')

        for img_id, labels in zip(df.image_id.values, df.labels.values):
            try:
                # can fail
                img = Image.open(os.path.join(folder, img_id))
                img.load()
                self.imgs.append(img)

                label_int = list(map(int, labels.replace('l', '').split(' ')))
                labels = torch.zeros(92)
                labels[label_int] = 1.0

                self.labels.append(labels)
            except:
                print(img_id, 'failed')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        if self.transform is None:
            return self.imgs[item], self.labels[item]
        else:
            return self.transform(self.imgs[item]), self.labels[item]


class TartuObjectModel(pl.LightningModule):
    def __init__(self, model, weights):
        super().__init__()
        model.train()
        self.model = model.features
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1280, 256),
            torch.nn.ReLU6(),
            torch.nn.Linear(256, 92),
            torch.nn.Sigmoid(),
        )
        self.weights = weights

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        convmap = self.model(x)
        features = torch.amax(convmap, dim=(2, 3))
        labels = self.classifier(features)
        # loss = F.mse_loss(labels, y)
        loss = F.cross_entropy(labels, y, weight=self.weights)
        self.log_dict({'loss': loss})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    transform = transforms.Compose([
        SquarePad(),
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomAffine(degrees=(-180, 180), scale=(0.8, 1.2), shear=(0,2,0,2))
    ]
    )
    dataset = TartuObjectDataset(transform=transform)

    w = dataset.labels[0]
    for lbls in dataset.labels[1:]:
        w += lbls
    w = w.sum()/w
    w = w.to('cuda:0')

    model = TartuObjectModel(
        torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT),
        weights=torch.ones(92).to('cuda:0')
    )
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=12)

    trainer = pl.Trainer(accelerator='gpu', devices='1', log_every_n_steps=1)
    trainer.fit(model=model, train_dataloaders=train_loader)
