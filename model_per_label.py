import numpy as np

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from jutils import img_exists, onehot, open_img_id, DEIT, CLIP, RN50, generate_split, train_enhance, labelstring

from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # read in images
    df = pd.read_csv('train_Kea.csv')
    # throw away missing images
    df = df.loc[df.image_id.apply(img_exists)].reset_index()
    df['Images'] = df['image_id'].apply(open_img_id)
    y = np.vstack(df['labels'].apply(onehot).values)

    models = []

    import torch
    import random
    torch.manual_seed(123)
    random.seed(123)
    from torchvision.transforms import AutoAugment
    augmenter = AutoAugment()
    features = DEIT()

    for i in range(92):
        negative = df.iloc[df.index[y[:, i] == 0]]
        positive = df.iloc[df.index[y[:, i] == 1]]
        diff = len(negative) - len(positive)
        iterations = diff // len(positive) + 1
        new_positives = []

        for _ in range(iterations):
            new_pos = positive.copy()
            new_pos['Images'] = new_pos['Images'].apply(augmenter)
            new_positives.append(new_pos)

        augmented = pd.concat([
            negative,
            positive,
            *new_positives
        ])

        Xs = features(augmented['Images'])
        ys = np.vstack(augmented['labels'].apply(onehot).values)
        model = LogisticRegression(random_state=1234)
        model.fit(Xs, ys[:,i])
        models.append(model)

    testdf = pd.read_csv('test.csv')
    testlabels = []
    labelsdf = pd.read_csv('labels.csv')
    for img_id in testdf.image_id:
        try:
            x = features([open_img_id(img_id)])
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



