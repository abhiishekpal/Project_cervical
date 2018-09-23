from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
from flask import Flask
import io
import cv2
import warnings
warnings.filterwarnings("ignore")
app = Flask(__name__)
model = None

def load_model():
	global model
	model = ResNet50(weights="imagenet")

def prepare_image(image, target):
	if image.mode != "RGB":
		image = image.convert("RGB")
	image = cv2.resize(image,target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)
	return image

@app.route("/predict", methods=["POST"])
def predict():
	data = {"success": False}
	if flask.request.method == "POST":
		if flask.request.files.get("image"):

			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			image = prepare_image(image, target=(224, 224))

			inp = np.ndarray(shape=(1,224,224,3),dtype=float)
			inp[0,:,:,:]=image
			preds = model.predict(inp)
			results = imagenet_utils.decode_predictions(preds)
			data["predictions"] = []

			for (imagenetID, label, prob) in results[0]:
				r = {"label": label, "probability": float(prob)}
				data["predictions"].append(r)

			data["success"] = True

	print(data)
	return flask.jsonify(data)

if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server...""please wait until server has fully started"))
	load_model()
	app.run(debug = True, threaded = False)
