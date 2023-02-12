import pandas as pd
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from sklearn import svm
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.filters import sobel
from scipy import ndimage as nd
from skimage.filters.rank import entropy
from skimage.morphology import disk
from imblearn.over_sampling import SMOTE
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import pandas as pd
from imblearn.combine import SMOTEENN
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import shannon_entropy
import entropy
from sklearn import metrics
import lightgbm as lgb
import re
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from skimage.feature import greycomatrix, greycoprops
import cv2
import copy
import os
import PIL
from skimage.io import imread
from skimage.transform import resize
import seaborn as sns
import pickle
from PIL import *
import cv2
from sklearn.utils import Bunch
import tensorflow as tf
from skimage.morphology import disk
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.python.keras import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
# from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import imgaug as ia
from pathlib import Path
import glob
from imblearn.combine import SMOTEENN
from sklearn import preprocessing
sample_sub = pd.read_csv('sample_submission.csv')
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    # iaa.Crop(percent=(0, 0.1)),  # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),

    # iaa.LinearContrast((0.75, 1.5)),
    # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # # iaa.Affine(
    #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    #     rotate=(-25, 25),
    #     shear=(-8, 8)
    # )
],
    random_order=True)
SIZE = 256
train_images = []
train_labels = []
for i in range(236):
    train_labels.append(0)
# print(len(train_labels))
for directory_path in glob.glob("output/train/EOSINOPHIL*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path,0)  # Reading color images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (SIZE, SIZE))  # Resize images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        train_images.append(img)
        # plt.imshow(img)
        # plt.show()
        for i in range(3):
            image_aug = seq(image=img)
            # plt.imshow(image_aug)
            # plt.show()
            train_images.append(image_aug)
        # train_labels.append(label)
# print(len(train_images))
# #236
for i in range(272):
    train_labels.append(1)
for directory_path in glob.glob("output/train/LYMPHOCYTE*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path,0)  # Reading color images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (SIZE, SIZE))  # Resize images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        train_images.append(img)
        for i in range(15):
            image_aug = seq(image=img)
            # plt.imshow(image_aug)
            # plt.show()
            train_images.append(image_aug)
        # train_labels.append(label)
# print(len(train_images))
# #187
for i in range(168):
    train_labels.append(2)
for directory_path in glob.glob("output/train/MONOCYTE*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path,0)  # Reading color images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (SIZE, SIZE))  # Resize images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        train_images.append(img)
        for i in range(13):
            image_aug = seq(image=img)
            # plt.imshow(image_aug)
            # plt.show()
            train_images.append(image_aug)
        # train_labels.append(label)
# print(len(train_images))
# # 168
for i in range(264):
    train_labels.append(3)
for directory_path in glob.glob("output/train/NEUTROPHIL*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path,0)  # Reading color images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (SIZE, SIZE))  # Resize images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        train_images.append(img)
        for i in range(1):
            image_aug = seq(image=img)
            # plt.imshow(image_aug)
            # plt.show()
            train_images.append(image_aug)
        # train_labels.append(label)
# print(len(train_images))
# # 264
# print(len(train_labels))
train_images = np.array(train_images)
train_labels = np.array(train_labels)
print(train_images.shape)
print(train_labels.shape)
x_train =train_images

val_images = []
val_labels = []
for i in range(14):
    val_labels.append(0)
# print(len(val_labels))
for directory_path in glob.glob("output/val/EOSINOPHIL*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path,0)  # Reading color images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (SIZE, SIZE))  # Resize images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        val_images.append(img)
print(len(train_images))
# # 264
# print(len(train_labels))
train_images = np.array(train_images)
train_labels = np.array(train_labels)
print(train_images.shape)
print(train_labels.shape)
x_train =train_images

val_images = []
val_labels = []
for i in range(14):
    val_labels.append(0)
# print(len(val_labels))
for directory_path in glob.glob("output/val/EOSINOPHIL*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path,0)  # Reading color images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (SIZE, SIZE))  # Resize images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        val_images.append(img)

        # train_labels.append(label)
# print(len(val_images))
# #236
for i in range(4):
    val_labels.append(1)
# print(len(val_labels))
for directory_path in glob.glob("output/val/LYMPHOCYTE*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path,0)  # Reading color images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (SIZE, SIZE))  # Resize images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        val_images.append(img)

        # train_labels.append(label)
print(len(val_images))
# #187
for i in range(3):
    val_labels.append(2)
# print(len(val_labels))
for directory_path in glob.glob("output/val/MONOCYTE*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path,0)  # Reading color images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (SIZE, SIZE))  # Resize images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        val_images.append(img)

# print(len(val_images))
# # 168
for i in range(33):
    val_labels.append(3)
# print(len(val_labels))

for directory_path in glob.glob("output/val/NEUTROPHIL*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path,0)  # Reading color images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (SIZE, SIZE))  # Resize images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        val_images.append(img)

val_images = np.array(val_images)
val_labels = np.array(val_labels)
print(len(val_images))
print(len(val_labels))

########################----------------------------------------------------------------------------------------
# df1 = pd.read_csv('train_set11.csv')
# train_labels = df1['Category']
# print(train_labels.head(15))
# le = preprocessing.LabelEncoder()
# train_labels_encoded = le.fit_transform(train_labels)
# print(train_labels_encoded)
########################----------------------------------------------------------------------------------------
def build_filters():
    filters = []
    ksize = 1  # Use size that makes sense to the image and fetaure size. Large may not be good.
    # On the synthetic image it is clear how ksize affects imgae (try 5 and 50)
    sigma = 3  # Large sigma on small features will fully miss the features.
    theta = 1 * np.pi / 4  # /4 shows horizontal 3/4 shows other horizontal. Try other contributions
    lamda = 2  # 1/4 works best for angled.
    gamma = 1  # Value of 1 defines spherical. Calue close to 0 has high aspect ratio
    # Value of 1, spherical may not be ideal as it picks up features from other regions.
    phi = 1  # Phase offset. I leave it to 0.
    kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
    kern /= 1.5 * kern.sum()
    filters.append(kern)
    return filters

def feature_extractor(img1, filters):
    accum = np.zeros_like(img1)
    for kern in filters:
        fimg = cv2.filter2D(img1, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
        return accum
########################----------------------------------------------------------------------------------------
print(train_images.shape)
print(val_images.shape)
print(train_labels)
# plt.plot(train_labels)
# plt.show()
filters=build_filters()
image_features = feature_extractor(train_images,filters)
image_features = np.expand_dims(image_features, axis=0)
X_for_RF = np.reshape(image_features, (train_images.shape[0], -1))
###

test_features = feature_extractor(val_images,filters)
test_features = np.expand_dims(test_features, axis=0)
test_for_RF = np.reshape(test_features, (val_images.shape[0], -1))
########################----------------------------------------------------------------------------------------
RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)
RF_model.fit(X_for_RF, train_labels)
# #
##
# SVM_model = svm.SVC(decision_function_shape='ovo')
# SVM_model.fit(X_for_RF, train_labels)
########################----------------------------------------------------------------------------------------

#Predict on evaluation
# Predict on test
test_prediction = RF_model.predict(test_for_RF)
#Inverse le transform to get original label back.
# test_prediction = le.inverse_transform(test_prediction)

#Print overall accuracy
from sklearn import metrics
print("Accuracy {0:.2f}%".format(100*accuracy_score(val_labels, test_prediction)))
print(confusion_matrix(val_labels, test_prediction))
print(classification_report(val_labels, test_prediction))
#Print confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(val_labels, test_prediction)

fig, ax = plt.subplots(figsize=(6,6))         # Sample figsize in inches
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, ax=ax)
plt.show()


########################----------------------------------------------------------------------------------------
vv =[]
df1 = pd.read_csv('train_set11.csv')
vv = df1['Category']
vv = np.array(vv)

plt.show()
print(len(vv))
le = preprocessing.LabelEncoder()
train_labels_encoded = le.fit_transform(vv)

########################----------------------------------------------------------------------------------------

test_images=[]
for directory_path in glob.glob("img2_test*"):
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (SIZE, SIZE))
        test_images.append(img)

test_images = np.array(test_images)

test_features = feature_extractor(test_images,filters)
test_features = np.expand_dims(test_features, axis=0)
tt1 = np.reshape(test_features, (test_images.shape[0], -1))

predictions = RF_model.predict(tt1)
pred_1 = le.inverse_transform(predictions)


sample_sub['Category'] = pred_1
sample_sub.to_csv('fixed_output__03.csv', index=False)
df22 = pd.read_csv('fixed_output__03.csv')
print(df22)