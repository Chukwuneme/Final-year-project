from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import joblib
import cv2

logisticRegr = joblib.load('c:/users/johnpaul/desktop/finalized_model.sav')


app = Flask(__name__,  static_folder='static')


@app.route("/skincancer", methods=["POST"])
def skincancer():
	if request.method == 'POST':
		uploaded_file = request.files['image']
		
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
			proba = logisticRegr.predict_proba[0][0]
			advice = "You need to see thee doctor asap"
		else:
			value = "malignant"
			proba = logisticRegr.predict_proba[0][1]
			advice = "Your cancer might be malignant. you need urgent medical attention"

		

	return render_template('secondpage.html', test=filenn, predictions = value, proba = proba, advice = advice)


@app.route("/index")
def home():
	return render_template('index.html',  test="static/default.png")
if __name__ == "__main__":
	app.run(debug=True)