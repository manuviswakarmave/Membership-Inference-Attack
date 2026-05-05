import os
import sys
import torch
import pandas as pd
import requests
import random
import argparse

from pathlib import Path
from torch.utils.data import Dataset, Subset
from torchvision.models import resnet18
import torchvision.transforms as transforms

from lira_implementation import get_stats, lira_attack, create_conf_csv, find_matching_shadow_model


# config
BASE = Path(__file__).parent
PUB_PATH = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
OUTPUT_CSV = BASE / "submission.csv"
PUB_MODEL_PATH = BASE / "pub_model.pt"


class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


# load datasets
print("Loading datasets...")
pub_ds = torch.load(PUB_PATH, weights_only=False)
priv_ds = torch.load(PRIV_PATH, weights_only=False)

##Class verification
print(type(pub_ds), type(priv_ds))


# normalization (same as training)
MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Normalize(mean=MEAN, std=STD),
])

pub_ds.transform = transform
priv_ds.transform = transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load model
print("Loading model...")
model = resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Linear(512, 9)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model = model.to(device)
model.eval()

cleaned_priv_ds = TaskDataset()
cleaned_priv_ds.ids = priv_ds.ids
cleaned_priv_ds.labels = priv_ds.labels
cleaned_priv_ds.imgs = priv_ds.imgs



for i in range(5):  # to increase the number of shadow models
    create_conf_csv(cleaned_priv_ds, device=device)
    lira_attack(model=model, dataset=cleaned_priv_ds, device=device, pub_test=False)