import numpy as np

import torch
import clip
from PIL import Image
from PIL.Image import Image as ImageType

import pandas as pd
import os
import torchvision.transforms.functional as fn
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.neighbors import KNeighborsClassifier


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

def hue_vector(imgs):

    vector = []
    for img in imgs:
        h = np.array(img.convert('HSV'))[:,:,0].ravel()
        counts = np.bincount(h, minlength=256)
        vector.append(counts/counts.sum())
    return np.array(vector)



if __name__ == '__main__':
    # read in images
    df = pd.read_csv('train.csv')
    # throw away missing images
    df = df.loc[df.image_id.apply(img_exists)]
    df = df.reset_index()

    df['Images'] = df['image_id'].apply(open_img_id)

    subsets = [
        set(np.arange(92)[onehot(lbl).astype(bool)])
        for lbl in df['labels']
    ]
    # subsets_used = set_cover(set(range(92)), subsets)


    def generate_splits(n, subsets):
        from itertools import permutations

        subsets_used = []
        splits = []
        for p, _ in zip(permutations(subsets), range(n)):
            subsets = set_cover(set(range(92)), p)

            val = df.index.isin(subsets)
            train = ~val

            train_idx = df.index[train]
            val_idx   = df.index[val]
            if subsets not in subsets_used:
                subsets_used.append(subsets_used)
                splits.append((train_idx, val_idx))

        return splits

    cv = generate_splits(10, subsets)
    print(len(cv))



    # read in labels
    labelsdf = pd.read_csv('labels.csv')
    labels = labelsdf['object'].values.tolist()
    clip_model = CLIP()
    details = pd.read_csv('labels_detailed.csv')['object']

    def features(imgs):
        # return np.hstack(
        #     (
        #         clip_model(imgs, details),
        #         clip_model.img_embeddings(imgs)
        #      )
        # )
        # return clip_model(imgs, details)
        return clip_model.img_embeddings(imgs)
    X = features(df['Images'])

    y = np.array([
        onehot(lbl) for lbl in df['labels']
    ]).astype(int)

    val_df = pd.read_csv('val_clean.csv')
    val_df['Images'] = val_df['image_id'].apply(open_img_id)
    # val_X = clip_model.img_embeddings(val_df['Images'])
    # val_X = rn50(val_df['Images'])
    val_X = features(df['Images'])

    val_y = np.array([
        onehot(lbl) for lbl in val_df['labels']
    ]).astype(int)

    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from ml_xgboost import MultiLabelXGBClassifier
    pipe = Pipeline(
        [
            ('sc', StandardScaler()),
            # ('pca', PCA(n_components=128, whiten=True)),
            # ('kbest', SelectKBest(k=512)),
            # ('model', MultiLabelXGBClassifier())
            # ('model', ClassifierChain(LogisticRegression(dual=True, solver='liblinear', fit_intercept=True, random_state=342985, class_weight='balanced')))
            ('model', ClassifierChain(KNeighborsClassifier(n_neighbors=3, algorithm='brute')))
        ])

    grid = GridSearchCV(
        pipe,
        param_grid={
            # 'model__base_estimator__C':[0.1, 0.5, 0.8, 1.0],
            # 'model__base_estimator__penalty': ['l2'],

            'model__base_estimator__metric': ['euclidean', 'cosine'],
            'model__base_estimator__weights': ['distance'],

            # 'model__n_estimators':[50],
            # 'model__booster':['gbtree'],
            # 'model__eta':[0.1, 0.3, 0.5],
            # 'model__lambda':[1.0, 2.0, 4.0],
            # 'model__alpha':[1.0, 2.0, 4.0],
            # 'model__max_delta_step': [1, 2, 4, 5,10],
            # 'model__objective':['binary:logistic'],
        },
        cv=cv,
        # scoring='f1_macro',
        scoring='f1_samples',
        # scoring='roc_auc_ovr_weighted',
        refit=True
    )
    grid.fit(X, y)

    model = grid.best_estimator_

    testdf = pd.read_csv('test.csv')
    testlabels = []

    # TODO: sometimes, we get NO predictions from the random forest. Investigate why.
    # TODO: try other classifiers like SVMs with the MultiClassifier strategy from sklearn.
    # TODO: class weights should be specified as the data is skewed as shit
    # TODO: contact the contact person as images are missing from both the training and test sets

    for img_id in testdf.image_id:
        try:
            # x = clip_model.img_embeddings(open_img_id(img_id))
            x = features([open_img_id(img_id)])
            prediction = model.predict(x)
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
    testdf.to_csv('joosep_submissions/knn.csv', index=False)
    print(grid.best_score_, grid.best_params_)





