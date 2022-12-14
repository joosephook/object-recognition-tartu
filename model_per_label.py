import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV


from sklearn.pipeline import Pipeline
import torch
import random

torch.manual_seed(123)
random.seed(123)
import torchvision.transforms as T
import torchvision.transforms.functional as fn
from jutils import img_exists, onehot, open_img_id,  labelstring, SquarePad, OpenCLIP

from sklearn.preprocessing import StandardScaler

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
    y_extra = np.vstack(extra['labels'].apply(onehot).values)

    models = []
    transforms = [
        fn.hflip,
        T.Compose([T.GaussianBlur(9)]),
        T.Compose([fn.hflip, T.GaussianBlur(9)]),
    ]
    features = OpenCLIP()
    scores = []

    augmentations = []

    for t in transforms:
        df2 = df.copy()
        df2['Images'] = df2['Images'].apply(t)
        extra2 = extra.copy()
        extra2['Images'] = extra2['Images'].apply(t)
        augmentations.append(df2)
        augmentations.append(extra2)

    augmentations = pd.concat([df, extra, *augmentations])
    Xs = features(augmentations['Images'])
    ys = np.vstack(augmentations['labels'].apply(onehot).values)

    for i in range(92):
        # create cross-validation folds
        original_positive = df.loc[(y[:,i]==1).ravel()]
        cv = []
        half = int(len(original_positive)/2)
        val = augmentations.index.isin(original_positive.index[0:half])
        train = ~val
        cv.append((augmentations.index[train], augmentations.index[val]))
        val = augmentations.index.isin(original_positive.index[half:])
        train = ~val
        cv.append((augmentations.index[train], augmentations.index[val]))

        model = LogisticRegressionCV(cv=cv)
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('model', model),
        ])
        param_grid = [
                dict(
                    model__solver=['liblinear'],
                    model__Cs=[np.linspace(0.1, 1.5, 10).tolist()],
                    model__max_iter=[400],
                    model__penalty=['l2'],
                    model__dual=[False, True],
                    model__class_weight=['balanced'],
                    model__random_state=[1234],
                    # model__n_jobs=[-1]
                     ),
                dict(
                    model__solver=['saga'],
                    model__Cs=[np.linspace(0.1, 1.5, 10).tolist()],
                    model__max_iter=[400],
                    model__penalty=['l1', 'l2'],
                    model__class_weight=['balanced'],
                    model__random_state=[1234],
                    # model__n_jobs=[-1]
                    ),
                dict(
                    model__solver=['saga'],
                    model__Cs=[np.linspace(0.1, 1.5, 10).tolist()],
                    model__l1_ratios=[np.linspace(0.1, 0.9, 10).tolist()],
                    model__max_iter=[400],
                    model__penalty=['elasticnet'],
                    model__class_weight=['balanced'],
                    model__random_state=[1234],
                    # model__n_jobs=[-1]
                    ),

                ]
        grid = GridSearchCV(
            pipe,
            param_grid,
            scoring='f1',
            n_jobs=-1,
            refit=True,
            cv=None
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
