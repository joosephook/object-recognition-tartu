from random_clip_forest import open_img_id, CLIP, onehot, img_exists, labelstring
import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('train.csv')
    df = df.loc[df.image_id.apply(img_exists)]
    df['Images'] = df['image_id'].apply(open_img_id)

    train_img_ids = df.image_id.values
    train = np.isin(np.array(os.listdir('images')), train_img_ids)
    labels = df['labels'].values
    labels_onehot = np.array([ onehot(lbl) for lbl in df['labels'] ]).astype(int)
    labelsdf = pd.read_csv('labels.csv')
    labels_human = np.array([
        labelsdf['object'].values[l.astype(bool)]
        for l in labels_onehot
    ], dtype=object)

    clip = CLIP()
    embeddings = clip.img_embeddings(df['Images'])
    print(embeddings.shape)
    np.savez('clip_embeddings_train.npz',
             img_id=train_img_ids,
             embeddings=embeddings,
             labels=labels,
             labels_onehot=labels_onehot,
             labels_human=labels_human
             )

    d = np.load('clip_embeddings_train.npz', allow_pickle=True)
    
    print(d.files)
    img_id        = d['img_id']
    embeddings    = d['embeddings']
    labels        = d['labels']
    labels_onehot = d['labels_onehot']
    labels_human  = d['labels_human']
    print(img_id[0], labels[0], labels_human[0])
    
    df = pd.read_csv('test.csv')
    df = df.loc[df.image_id.apply(img_exists)]
    df['Images'] = df['image_id'].apply(open_img_id)

    test_img_ids = df.image_id.values
    embeddings = clip.img_embeddings(df['Images'])
    
    np.savez('clip_embeddings_test.npz',
             img_id=test_img_ids,
             embeddings=embeddings,
    )
    
    train = np.load('clip_embeddings_train.npz')
    X_train = train['embeddings'] 
    y_train = train['labels_onehot']
    

    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score
    from sklearn.linear_model import LogisticRegression

    rf = MultiOutputClassifier(SVC(C=2.0, class_weight='balanced', random_state=403895))
    rf.fit(X_train, y_train)
    print(f'f1 score: {f1_score(y_train, rf.predict(X_train), average="macro"):.3f}')
    print(rf.score(X_train, y_train))

    test = np.load('clip_embeddings_test.npz', allow_pickle=True)
    X_test = test['embeddings']
    test_img_ids = test['img_id']

    testdf = pd.read_csv('test.csv')
    testlabels = []

    for img_id in testdf['image_id']:
        if img_id in test_img_ids:
            prediction = rf.predict(test['embeddings'][img_id == test_img_ids].reshape(1,-1))
            predicted_labels = labelstring(prediction.astype(bool))
            if len(predicted_labels) == 0:
                testlabels.append('l1')
            else:
                testlabels.append(predicted_labels)
            print('='*40)
            print(img_id)
            print(labelsdf.loc[labelsdf.label_id.isin(testlabels[-1].split(' ')), 'object'])
            print('='*40)
        else:
            # default label for the missing images in our test set
            print('test image', img_id, 'missing from images')
            testlabels.append('l0')


    testdf['labels'] = testlabels
    testdf.to_csv('joosep_submissions/clip_svc_classifier.csv', index=False)
    
    
    


    
    




