# import python libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# initailes flask app
app = Flask(__name__)
# load model pickel file
model = pickle.load(open('model.pkl', 'rb'))

# create route index.html
@app.route('/')
def home():
    return render_template('index.html')

# create route for predict classification
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # read json data which pass from front end
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    # predict model
    prediction = model.predict(final_features)
    # save output variable for output variable
    output = prediction[0]

    return render_template('index.html', prediction_text='prefrence classification $ {}'.format(output))

# create route for API predict classification
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    # read json data which pass from front end
    data = request.get_json(force=True)
    # predict model
    prediction = model.predict([np.array(list(data.values()))])
    # save output variable for output variable
    output = prediction[0]
    # retrun output as json data
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)