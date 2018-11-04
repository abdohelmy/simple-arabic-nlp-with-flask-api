from flask import Flask
from flask import jsonify,request
from LR import Model
clf = Model()

app = Flask(__name__)

@app.route("/read")
def read():
	clf.read_df("C:\\Users\\DELL\\Desktop\\Social_Network_Ads.csv")
	return clf.dataset.head().to_json()

@app.route("/split")
def split():
	clf.split_df()
	return "Data split Done!"

@app.route("/scale")
def scale():
	clf.scaling()
	return "scaling Done!"

@app.route("/train_test")
def train_test():
	clf.train_test(0.25)
	return "train_test Done!"

@app.route("/train")
def train():
	clf.train()
	return "Training done!"

@app.route("/evaluate")
def evaluate():
	score = clf.evaluate()
	resp = {"score":score}
	return jsonify(resp)

@app.route("/predict",methods=["GET"])
def predict():
	age = request.args.get('age')
	salary = request.args.get('salary')
	y_pred = clf.predict([age,salary])
	resp = {"class":int(y_pred[0])}
	return jsonify(resp)


if __name__ == '__main__':
	try:
		app.run(port='9090',host='0.0.0.0')
	except Exception as e:
		print("Error")