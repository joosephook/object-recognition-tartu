import os

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from PIL.Image import Image as ImageType
from torchvision import transforms as T
from torchvision.transforms import functional as FT



def img_exists(img_id):
    exists = os.path.isfile(os.path.join('images', img_id))
    return exists


def onehot(labels: str) -> np.ndarray:
    labels_int = list(map(int, labels.replace('l', '').split(' ')))
    labels = np.zeros(92, dtype=int)
    labels[labels_int] = 1
    return labels


def labelstring(onehot: np.ndarray) -> str:
    labels = np.array([f'l{i}' for i in range(92)])
    return ' '.join(labels[onehot[0]])


from more_itertools import chunked
class BEIT:
    def __init__(self):
        from transformers import BeitFeatureExtractor, BeitForImageClassification
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
        self.model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k').to(self.device)
    def __call__(self, images):
        features = []
        for imgs in chunked(images, 100):
            inputs = self.feature_extractor(images=imgs, return_tensors="pt").to(self.device)
            with torch.no_grad():
                features.append(self.model(**inputs)['logits'].cpu().numpy())
        return np.vstack(features)


from skimage.feature import hog
from skimage import data, exposure

class HOG:
    def __call__(self, images):
        features = [
            hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), channel_axis=-1, feature_vector=True)
            for image in images
        ]
        return np.vstack(features)

class CLIP:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def __call__(self, imgs: [ImageType]):
        if isinstance(imgs, ImageType):
            imgs = [imgs]

        images = torch.cat([self.preprocess(img).unsqueeze(0).to(self.device) for img in imgs], dim=0)
        with torch.no_grad():
            img_embeddings = self.model.encode_image(images.to(self.device))
        return img_embeddings.cpu().numpy()
    
    def text_similarity(self, imgs: [ImageType], texts: [str]):
        texts = clip.tokenize(texts).to(self.device)
        images = torch.cat([self.preprocess(img).unsqueeze(0).to(self.device) for img in imgs], dim=0)
        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(images, texts)
        return logits_per_image.cpu().numpy()


def open_img_id(img_id: str) -> ImageType:
    img = Image.open(os.path.join('images', img_id))
    img.load()
    return img


from torchvision.models import resnet50, ResNet50_Weights
class RN50:
    def __init__(self):
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=weights)
        self.model.eval()
        self.t = weights.transforms()

    def __call__(self, x):
        x = torch.cat([self.t(img).unsqueeze(0) for img in x], dim=0)
        with torch.no_grad():
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            return torch.sum(x, dim=(2, 3)).cpu().numpy()


def set_cover(universe, subsets):
    """Find a family of subsets that covers the universal set"""
    elements = set(e for s in subsets for e in s)
    # Check the subsets cover the universe
    if elements != universe:
        return None
    covered = set()
    cover = []
    # Greedily add the subsets with the most uncovered points

    used_subsets = []
    while covered != elements:
        dists = []
        for i, subset in enumerate(subsets):
            dists.append((len(subset - covered), i))

        d, i = max(dists)
        cover.append(subsets[i])
        used_subsets.append(i)
        covered |= subsets[i]

    return used_subsets

class DEIT:
    def __init__(self):
        from deit.models_v2 import deit_large_patch16_LS
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.DEIT = deit_large_patch16_LS(pretrained_21k=True, pretrained=True, img_size=384).to(self.device)
        self.DEIT.eval()
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        self.DEIT_T = T.Compose([T.Resize(400), T.CenterCrop(384), T.ToTensor(),
                            T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                            lambda x: x.unsqueeze(0)])

    def __call__(self, imgs):
        with torch.no_grad():
            features = []
            for img in imgs:
                features.append(self.DEIT.forward_features(self.DEIT_T(img).to(self.device)).cpu().numpy())
            return np.concatenate(features, axis=0)

import open_clip
class OpenCLIP:
    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def __call__(self, imgs: [ImageType]):
        if isinstance(imgs, ImageType):
            imgs = [imgs]

        features = []
        with torch.no_grad():
            for imgs in chunked(imgs, 100):
                images = torch.cat([self.preprocess(img).unsqueeze(0).to(self.device) for img in imgs], dim=0)
                img_embeddings = self.model.encode_image(images.to(self.device)).cpu().numpy()
                features.append(img_embeddings)
        return np.vstack(features)


def generate_split(df):
    y_oh = np.array([onehot(lbl) for lbl in df['labels']]).astype(int)
    subsets_available = [
        set(np.arange(92)[lbl.astype(bool)]) for lbl in y_oh
    ]
    subsets = set_cover(set(range(92)), subsets_available)

    val = df.index.isin(subsets)
    train = ~val

    train_idx = df.index[train]
    val_idx = df.index[val]

    yval = y_oh[val_idx]
    ytrain = y_oh[train_idx]

    val_count = yval.sum(axis=0)
    train_count = ytrain.sum(axis=0)


    assert np.all(val_count > 0) and np.all(train_count > 0)

    return [(train_idx, val_idx)]


def label_test_set(m, features):
    testdf = pd.read_csv('test.csv')
    testlabels = []
    labelsdf = pd.read_csv('labels.csv')
    for img_id in testdf.image_id:
        try:
            x = features([open_img_id(img_id)])
            prediction = m.predict(x)
            predicted_labels = labelstring(prediction.astype(bool))

            if len(predicted_labels) == 0:
                testlabels.append('l1')
            else:
                testlabels.append(predicted_labels)
            print(img_id,
                  ' '.join(labelsdf.loc[labelsdf.label_id.isin(testlabels[-1].split(' ')), 'object'].values.ravel()),
                  sep='\t')
        except FileNotFoundError:
            print(img_id, 'missing, defaulting to l0')
            testlabels.append('l0')

    testdf['labels'] = testlabels
    import time
    timestamp = int(time.time())
    import os
    import shutil
    outdir = f'joosep_submissions/{timestamp}'
    os.mkdir(outdir)
    testdf.to_csv(os.path.join(outdir, f'{timestamp}_test.csv'), index=False)
    shutil.copy(__file__, outdir)


def train_enhance(dataframe):
    transforms = [FT.hflip]

    for t in transforms:
        df2 = dataframe.copy()
        df2['Images'] = df2['Images'].apply(t)
        dataframe = pd.concat((dataframe, df2), axis=0)
    return dataframe

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return FT.pad(image, padding, 0, 'constant')

