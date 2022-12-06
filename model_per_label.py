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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline
import torch
import random

torch.manual_seed(123)
random.seed(123)
import torchvision.transforms as T
import torchvision.transforms.functional as fn
from jutils import img_exists, onehot, open_img_id, DEIT, CLIP, RN50, generate_split, train_enhance, labelstring, BEIT, \
    SquarePad, HOG, OpenCLIP

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    labelsdf = pd.read_csv('labels.csv')
    padder = SquarePad()
    df = pd.read_csv('train_fixed.csv')
    df = df.loc[df.image_id.apply(img_exists)].reset_index()
    df['Images'] = df['image_id'].apply(open_img_id).apply(padder)
    y = np.vstack(df['labels'].apply(onehot).values)

    extra = pd.read_csv('train_Kea.csv')[['image_id', 'labels']]
    extra = extra.loc[extra.image_id.apply(img_exists)].reset_index()
    extra['Images'] = extra['image_id'].apply(open_img_id).apply(padder)
    extra = extra.loc[~extra.image_id.isin(df.image_id)]
    extra = extra.iloc[0:1]
    y_extra = np.vstack(extra['labels'].apply(onehot).values)

    models = []
    transforms = [
        fn.hflip,
        T.Compose([T.GaussianBlur(9)]),
        T.Compose([fn.hflip, T.GaussianBlur(9)]),
    ]
    features = OpenCLIP()
    scores = []
    for i in range(92):
        # augment positive examples of a class
        positive = pd.concat([
            df.loc[(y[:, i] == 1).ravel()],
            extra.loc[(y_extra[:, i] == 0).ravel()],
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

        # create cross-validation folds
        original_positive = df.loc[(y[:,i]==1).ravel()]
        cv = []
        half = int(len(original_positive)/2)
        val = augmented.index.isin(original_positive.index[0:half])
        train = ~val
        cv.append((augmented.index[train], augmented.index[val]))
        val = augmented.index.isin(original_positive.index[half:])
        train = ~val
        cv.append((augmented.index[train], augmented.index[val]))

        Xs = features(augmented['Images'].tolist())
        ys = np.vstack(augmented['labels'].apply(onehot).values)
        model = LogisticRegression()
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('model', model)
        ])
        param_grid = [
                dict(
                    model__solver=['liblinear'],
                    model__C=[0.8, 0.9, 1.0],
                    model__max_iter=[200],
                    model__penalty=['l2'],
                    model__dual=[False, True],
                    model__class_weight=['balanced'],
                    model__random_state=[1234],
                     ),
                dict(
                    model__solver=['saga'],
                    model__C=[0.8, 0.9, 1.0],
                    model__max_iter=[200],
                    model__penalty=['l1', 'l2'],
                    model__class_weight=['balanced'],
                    model__n_jobs=[-1]
                    ),
                ]
        grid = GridSearchCV(
            pipe,
            param_grid,
            scoring='f1',
            n_jobs=-1,
            refit=True,
            cv=cv
            )
        grid.fit(Xs, ys[:, i])
        print(labelsdf.iloc[i], grid.cv_results_['mean_test_score'])
        grids = [
           grid,
        ]
        grid_scores = [
            g.best_score_ for g in grids
        ]
        best = np.argmax(grid_scores)
        models.append(grids[best].best_estimator_)
        scores.append(grids[best].best_score_)


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
    import joblib
    for i, m in enumerate(models):
        joblib.dump(m, outdir+f'/l{i}.pkl')
