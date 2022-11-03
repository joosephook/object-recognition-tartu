import numpy as np
from sklearn.ensemble import RandomForestClassifier

import torch
import clip
from PIL import Image
from PIL.Image import Image as ImageType

import pandas as pd
import os
import torchvision.transforms.functional as fn

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return fn.pad(image, padding, 0, 'constant')

def img_exists(img_id):
    exists = os.path.isfile(os.path.join('images', img_id))
    if not exists:
        print(img_id)
    return exists

def onehot(labels: str) -> np.ndarray:
    labels_int = list(map(int, labels.replace('l', '').split(' ')))
    labels = np.zeros(92)
    labels[labels_int] = 1.0
    return labels

def labelstring(onehot: np.ndarray) -> str:
    labels = np.array([f'l{i}' for i in range(92)])
    return ' '.join(labels[onehot[0]])


class CLIP:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def __call__(self, imgs: [ImageType], texts: [str]):
        texts = clip.tokenize(texts).to(self.device)
        images = torch.cat([self.preprocess(img).unsqueeze(0).to(self.device) for img in imgs], dim=0)
        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(images, texts)
        return logits_per_image.cpu().numpy()

    def img_embeddings(self, imgs: [ImageType]):
        if isinstance(imgs, ImageType):
            imgs = [imgs]

        images = torch.cat([self.preprocess(img).unsqueeze(0).to(self.device) for img in imgs], dim=0)
        with torch.no_grad():
            img_embeddings = self.model.encode_image(images)
        return img_embeddings.cpu().numpy()

def open_img_id(img_id: str) -> ImageType:
    img = Image.open(os.path.join('images', img_id))
    img.load()
    return img


def resnet50_features():
    from torchvision.models import resnet50, ResNet50_Weights
    weights = ResNet50_Weights.IMAGENET1K_V2

    model = resnet50(weights=weights)
    model.eval()

    t = weights.transforms()

    def features(x):
        # See note [TorchScript super()]
        x = torch.cat([t(img).unsqueeze(0) for img in x], dim=0)
        with torch.no_grad():
            x = weights.transforms()(x)
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)

            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            return torch.sum(x, dim=(2,3)).cpu().numpy()

    return features



if __name__ == '__main__':
    # read in images
    df = pd.read_csv('train.csv')
    # throw away missing images
    df = df.loc[df.image_id.apply(img_exists)]
    df['Images'] = df['image_id'].apply(open_img_id)
    # pad all images to a square to not lose information
    square_pad = SquarePad()
    # df['Images'] = df['Images'].apply(square_pad)


    print(df.shape)
    print(df.shape)

    # read in labels
    labelsdf = pd.read_csv('labels.csv')
    labels = labelsdf['object'].values.tolist()
    clip_model = CLIP()

    img_embeddings = clip_model.img_embeddings(df['Images'])
    X = (img_embeddings + clip_model.img_embeddings(df['Images'].apply(fn.hflip)))/2

    y = np.array([
        onehot(lbl) for lbl in df['labels']
    ]).astype(int)

    from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    order = pd.read_csv('labels.csv')['order']
    sc = StandardScaler()
    sc.fit(X)
    rf = ClassifierChain(LogisticRegression(dual=True, C=0.1, solver='liblinear', class_weight='balanced', random_state=3425), order=order)
    X = sc.transform(X)
    rf.fit(X, y)

    testdf = pd.read_csv('test.csv')
    testlabels = []

    # TODO: sometimes, we get NO predictions from the random forest. Investigate why.
    # TODO: try other classifiers like SVMs with the MultiClassifier strategy from sklearn.
    # TODO: class weights should be specified as the data is skewed as shit
    # TODO: contact the contact person as images are missing from both the training and test sets

    for img_id in testdf.image_id:
        try:
            x1 = clip_model.img_embeddings(open_img_id(img_id))
            x3 = clip_model.img_embeddings(fn.hflip(open_img_id(img_id)))
            x = (x1+x3)/2
            x = sc.transform(x)
            prediction = rf.predict(x)
            predicted_labels = labelstring(prediction.astype(bool))

            if len(predicted_labels) == 0:
                testlabels.append('l1')
            else:
                testlabels.append(predicted_labels)
            print('='*40)
            print(img_id)
            print(labelsdf.loc[labelsdf.label_id.isin(testlabels[-1].split(' ')), 'object'])
            print('='*40)
        except FileNotFoundError:
            print(img_id)
            testlabels.append('l0')
    testdf['labels'] = testlabels
    testdf.to_csv('joosep_submissions/clip_logistic_dual_scaled_hflip_merge_strong_reg_scaled.csv', index=False)





