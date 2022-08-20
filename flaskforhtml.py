from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import joblib


logisticRegr = joblib.load('c:/users/johnpaul/desktop/finalized_model.sav')
#predictions2 = logisticRegr.predict(test_files)

app = Flask(__name__,  static_folder='static')


@app.route("/skincancer", methods=["POST"])
def skincancer():
	if request.method == 'POST':
		uploaded_file = request.files['image']
		uploaded_file.filename != ''
		#os.remove("default.png")
		num = 1
		num = num + 1
		name = "default" + str(num)
		filenn = "static/" + name + ".png"
		uploaded_file.save(filenn)
		image = cv2.imread(filenn)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		ggg = "greay"
		hhh = "static/" + ggg + ".png"
		cv2.imwrite(hhh, image)
		image = cv2.imread(hhh)
		image = cv2.resize(image, (440, 440))
		image_flattened = image.flatten()
		flattened = []
		flattened.append(image_flattened)
		flattened = np.array(flattened)
		predictions2 = logisticRegr.predict(flattened)
		if predictions2[0] == 0:
			value = "benign"
		else:
			value = "malignant"
	return render_template('secondpage.html', test=filenn, predictions = value)


@app.route("/index")
def home():
	return render_template('index.html',  test="static/default.png")
if __name__ == "__main__":
	app.run(debug=True)