import pickle
import numpy as np

from flask import Flask,render_template, request,url_for,jsonify

app = Flask(__name__)

with open('classifier.pkl', 'rb') as file :
        model = pickle.load(file)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():
    
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    if output == 1 :
        return render_template('index.html',prediction_text="the person is likely to have Diabetes")
    else:
        return render_template('index.html',prediction_text="the person is not likely to have Diabetes")

if __name__ == "__main__":
    app.run(debug=True)