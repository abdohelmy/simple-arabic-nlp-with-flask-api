from flask import Flask
from flask import jsonify,request
from model import Model
from flask import Markup
clf = Model()

app = Flask(__name__)


@app.route("/train",methods=["GET","POST"])
def train():
    classy=request.values.get('model')
    classification_report=clf.train(classy)
	
    return  "<form method=get action=http://127.0.0.1:9090/predict>\
	<table style='width:100%'>\
  <tr>\
  <th>'class'</th>\
    <th>precision</th>\
    <th>recall</th> \
    <th>f1-score</th>\
  </tr>\
  <tr>\
    <th>'0'</th>\
    <th>"+str(classification_report[70:80])+"</th>\
    <th>"+str(classification_report[80:90])+"</th>\
    <th>"+str(classification_report[90:100])+"</th>\
  </tr>\
  <tr>\
    <th>'1'</th>\
    <th>"+str(classification_report[120:130])+"</th>\
    <th>"+str(classification_report[130:140])+"</th>\
    <th>"+str(classification_report[140:150])+"</th>\
  </tr>\
</table>\
\
<form method=get action=http://127.0.0.1:9090/predict>\
\
    <input type=text   name='text'>\
<input type=submit value='predict'>\
\
</form>\
\
</form>"
    

@app.route("/evaluate")
def evaluate():
	score = clf.evaluate()
	resp = {"score":score}
	return jsonify(resp)

@app.route("/predict",methods=["GET","POST"])
def predict():
    text2 = request.args.get('text')
    y_pred = clf.predict(text2)
    resp = {"class":int(y_pred[0])}
    return jsonify(resp)


if __name__ == '__main__':
	try:
		app.run(port=9090,host='0.0.0.0')
	except Exception as e:
		print("Error")