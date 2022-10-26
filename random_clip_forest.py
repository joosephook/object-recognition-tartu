import numpy as np
from sklearn.ensemble import RandomForestClassifier

import torch
import clip
from PIL import Image

import pandas as pd
import os

def img_exists(img_id):
    return os.path.isfile(os.path.join('images', img_id))

def onehot(labels: str) -> np.ndarray:
    labels_int = list(map(int, labels.replace('l', '').split(' ')))
    labels = np.zeros(92)
    labels[labels_int] = 1.0
    return labels

def labelstring(onehot: np.ndarray) -> str:
    labels = np.array([f'l{i}' for i in range(92)])
    print(onehot.shape)
    return ' '.join(labels[onehot[0]])


if __name__ == '__main__':
    # read in images
    df = pd.read_csv('train.csv')
    # throw away missing images
    df = df.loc[df.image_id.apply(img_exists)]
    imgs = []
    for img in df['image_id']:
        img = Image.open(os.path.join('images', img))
        img.load()
        imgs.append(img)


    # read in labels
    labels = pd.read_csv('labels.csv')['object'].values.tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    texts = clip.tokenize(labels).to(device)

    images = torch.cat([preprocess(img).unsqueeze(0).to(device) for img in imgs], dim=0)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(images, texts)

    X = logits_per_image.cpu().numpy()
    y = np.array([
        onehot(lbl) for lbl in df['labels']
    ]).astype(int)

    rf = RandomForestClassifier(n_estimators=111, max_features=None, random_state=345)
    rf.fit(X, y)

    testdf = pd.read_csv('test.csv')
    testlabels = []

    # TODO: sometimes, we get NO predictions from the random forest. Investigate why.
    # TODO: try other classifiers like SVMs with the MultiClassifier strategy from sklearn.
    # TODO: class weights should be specified as the data is skewed as shit
    # TODO: contact the contact person as images are missing from both the training and test sets

    for img_id in testdf.image_id:
        try:
            img = Image.open(os.path.join('images', img_id))
            image = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits_per_image, logits_per_text = model(image, texts)
            prediction = rf.predict(logits_per_image.cpu().numpy())
            # assert np.any(prediction > 0)
            predicted_labels = labelstring(prediction.astype(bool))
            if len(predicted_labels) == 0:
                testlabels.append('l1')
            else:
                testlabels.append(predicted_labels)
            print(testlabels)
        except FileNotFoundError:
            testlabels.append('l0')
    testdf['labels'] = testlabels
    testdf.to_csv('joosep_submissions/clip_random_forest_initial.csv', index=False)





