import pickle
import os

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1-preparing data
input_dir = 'C:/Users/poyra/OneDrive/Documents/Visual Studio Code/Python/CYZ-Eğitim/Image Classification - Cats and Dogs/data/test_set/test_set'
categories = ['cats', 'dogs']

data = []
labels = []
for category_idx, category in enumerate(categories):
    category_dir = os.path.join(input_dir, category)
    for file in os.listdir(category_dir):
        if file.endswith(('.jpg', '.jpeg', '.png')):  # Check if file is an image
            img_path = os.path.join(category_dir, file)
            img = imread(img_path)
            img = resize(img, (15, 15))
            data.append(img.flatten())
            labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)
print("Hello")

# 2-train/test split

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle = True, stratify=labels)

# 3-train classifier

classifier = SVC()

parameters = [{'gamma': [0.1, 0.01, 0.001], 'C':[1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

# test performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score * 100)))

pickle.dump(best_estimator, open('./model.p', 'wb'))