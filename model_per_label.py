import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import ClassifierChain
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
import torch
import random

torch.manual_seed(123)
random.seed(123)
import torchvision.transforms as T
import torchvision.transforms.functional as fn
from jutils import img_exists, onehot, open_img_id, DEIT, CLIP, RN50, generate_split, train_enhance, labelstring, BEIT, \
    SquarePad, HOG

from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    padder = SquarePad()
    df = pd.read_csv('train_fixed.csv')
    df = df.loc[df.image_id.apply(img_exists)].reset_index()
    df['Images'] = df['image_id'].apply(open_img_id).apply(padder)
    y = np.vstack(df['labels'].apply(onehot).values)

    extra = pd.read_csv('train_Kea.csv')[['image_id', 'labels']]
    extra = extra.loc[extra.image_id.apply(img_exists)].reset_index()
    extra['Images'] = extra['image_id'].apply(open_img_id).apply(padder)
    extra = extra.iloc[~extra.index.isin(df.index)]
    y_extra = np.vstack(extra['labels'].apply(onehot).values)

    models = []

    transforms = [
        fn.hflip,
        T.Compose([T.GaussianBlur(9)]),
        T.Compose([fn.hflip, T.GaussianBlur(9)]),
    ]
    features = CLIP()

    cv_base = generate_split(df)
    scores = []
    for i in range(92):
        positive = pd.concat([
            df.loc[y[:, i] == 1],
            extra.loc[y_extra[:, i] == 1],
        ])

        new_positives = []

        for t in transforms:
            new_pos = positive.copy()
            new_pos['Images'] = new_pos['Images'].apply(t)
            new_positives.append(new_pos)

        augmented = pd.concat([
            df,
            extra,
            *new_positives
        ])
        # create new cv set by including the samples we picked for validation
        val = augmented.index.isin(cv_base[0][1])
        train = ~val
        train_idx = augmented.index[train]
        val_idx = augmented.index[val]
        cv = [(train_idx, val_idx)]

        Xs = features(augmented['Images'].tolist())
        ys = np.vstack(augmented['labels'].apply(onehot).values)
        model = LogisticRegression(solver='saga', class_weight='balanced', random_state=1234)

        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('model', model)
        ])
        grid = GridSearchCV(pipe,
                            dict(
                                model__C=[0.8, 0.9, 1.0],
                                model__max_iter=[100, 500, 1000],
                                model__penalty=['l2', 'l1']
                            ),
                            scoring='f1',
                            n_jobs=-1, refit=True, cv=cv)
        grid.fit(Xs, ys[:, i])
        models.append(grid.best_estimator_)
        scores.append(grid.best_score_)

    testdf = pd.read_csv('test.csv')
    testlabels = []
    labelsdf = pd.read_csv('labels.csv')
    for lbl, score in zip(labelsdf['object'], scores):
        print(lbl, score)
    total_predictions = np.zeros(92)
    for img_id in testdf.image_id:
        try:
            x = features([padder(open_img_id(img_id))])
            prediction = []
            for m in models:
                prediction.append(
                    m.predict(x)
                )
            prediction = np.array(prediction).reshape(1, -1)
            total_predictions += prediction[0]
            predicted_labels = labelstring(prediction.astype(bool))

            if len(predicted_labels) == 0:
                testlabels.append('l68 l77 l78 l8')
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
    print(np.mean(scores))
    print('missing predictions for labels:\n', labelsdf.loc[total_predictions == 0, 'object'])
