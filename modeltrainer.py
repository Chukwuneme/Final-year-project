import os
import cv2
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

benign = "./dataset/train/benign"
malignant = "./dataset/train/malignant"

images = []
target = []


v_benign = "./dataset/validation/benign"
v_malignant = "./dataset/validation/malignant"

validation_images = []
validation_target = []

for b, m in zip(os.listdir(benign), os.listdir(malignant)):
	path = os.path.join(benign, b)
	pic = cv2.imread(path)
	pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
	pic = cv2.resize(pic, (440, 440))
	pic = pic.flatten()
	images.append(pic)
	target.append(0)

	path = os.path.join(malignant, m)
	pic = cv2.imread(path)
	pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
	pic = cv2.resize(pic, (440, 440))
	pic = pic.flatten()
	images.append(pic)
	target.append(1)


scaler = MinMaxScaler()
images = scaler.fit_transform(images)


model = LogisticRegression()
model.fit(images, target)


for b, m in zip(os.listdir(v_benign), os.listdir(v_malignant)):
	path = os.path.join(v_benign, b)
	pic = cv2.imread(path)
	pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
	pic = cv2.resize(pic, (440, 440))
	pic = pic.flatten()
	validation_images.append(pic)
	validation_target.append(0)

	path = os.path.join(v_malignant, m)
	pic = cv2.imread(path)
	pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
	pic = cv2.resize(pic, (440, 440))
	pic = pic.flatten()
	validation_images.append(pic)
	validation_target.append(1)

scaler = MinMaxScaler()
images = scaler.fit_transform(validation_images)

pred = model.predict(validation_images)
print(pred)
print("ghjk")
print(accuracy_score(validation_target, pred))








print(np.array(images).shape, len(target))