import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import ClassifierChain
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from xgboost import XGBClassifier

from jutils import img_exists, onehot, open_img_id, DEIT, CLIP, RN50, generate_split, train_enhance, labelstring, BEIT, SquarePad

from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    padder  = SquarePad()
    # read in images
    df = pd.read_csv('train.csv')
    # throw away missing images
    df = df.loc[df.image_id.apply(img_exists)].reset_index()
    df['Images'] = df['image_id'].apply(open_img_id).apply(padder)
    y = np.vstack(df['labels'].apply(onehot).values)

    models = []

    import torch
    import random
    torch.manual_seed(123)
    random.seed(123)
    augmenter = AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
    import torchvision.transforms as T
    import torchvision.transforms.functional as fn
    transforms = [
        fn.hflip,
        T.Compose([T.GaussianBlur(9)]),
        T.Compose([fn.hflip, T.GaussianBlur(9)]),
        T.Compose([fn.hflip, T.RandomAdjustSharpness(0.1, p=1)]),
    ]
    features = CLIP()

    cv = generate_split(df)
    train, val = cv[0]

    scores = []
    for i in range(92):
        negative = df.iloc[df.index[y[:, i] == 0]]
        positive = df.iloc[df.index[y[:, i] == 1]]
        diff = len(negative) - len(positive)
        iterations = diff // len(positive) + 1
        new_positives = []

        # for t in transforms:
        for _ in range(iterations):
            new_pos = positive.copy()
            new_pos['Images'] = new_pos['Images'].apply(augmenter)
            new_positives.append(new_pos)

        augmented = pd.concat([
            negative,
            positive,
            *new_positives
        ])

        train = augmented.index.isin(cv[0][0])
        val = augmented.index.isin(cv[0][1])
        augmented.reset_index(inplace=True)
        train_idx = augmented.index[train]
        val_idx = augmented.index[val]
        cv = [(train_idx, val_idx)]

        Xs = features(augmented['Images'].tolist())
        ys = np.vstack(augmented['labels'].apply(onehot).values)
        model = LogisticRegression(solver='saga', max_iter=400, class_weight='balanced', random_state=1234)
        grid = GridSearchCV(
            model,
            param_grid={
                'C':[0.7, 0.8, 0.9, 1.0],
                # 'penalty': ['l2', 'l1'],
            },
            cv=cv,
            scoring='f1',
            refit=True
        )
        grid.fit(Xs, ys[:,i])
        models.append(grid.best_estimator_)
        scores.append(grid.best_score_)

    testdf = pd.read_csv('test.csv')
    testlabels = []
    labelsdf = pd.read_csv('labels.csv')
    for img_id in testdf.image_id:
        try:
            x = features([padder(open_img_id(img_id))])
            prediction = []
            for m in models:
                prediction.append(
                    m.predict(x)
                )
            prediction = np.array(prediction).reshape(1, -1)
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
    import matplotlib.pyplot as plt

    label_name = []
    with open('labels.csv', 'r') as f:
        f.readline()
        for line in f:
            label_name.append(line.strip().split(',')[1])

    zero_fscores = (np.asarray(scores) == 0).sum()
    plt.gcf().set_size_inches(10, 20)
    plt.barh(np.arange(92), scores)
    plt.yticks(np.arange(92), label_name);
    plt.title(str(grid.best_estimator_) + f'\nnumber of zero F-scores: {zero_fscores}')
    plt.show()
    print('number of zero F-scores', )




