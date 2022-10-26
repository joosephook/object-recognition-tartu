from tensorflow.keras.applications.resnet50 import ResNet50,  preprocess_input
from tensorflow.keras.preprocessing import image

import numpy as np

model = ResNet50(weights='imagenet', include_top=False)

img_path = 'images/img296.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# toores, liigitamata tensor otse n2rviv6rgust
conv_featuremap = model.predict(x)
print(conv_featuremap.shape)

# 1. variant: v6tame maksimaalse v22rtuse igas 7x7 maatriksis
features = conv_featuremap.max(axis=(0,1,2))
print(features.shape)

# 2. variant: v6tame keskmise v22rtuse
features = conv_featuremap.mean(axis=(0,1,2))
print(features.shape)

# 3. variant: liidame k6ik v22rtused kokku
features = conv_featuremap.sum(axis=(0,1,2))
print(features.shape)