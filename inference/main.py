import flask
import joblib
import numpy as np


app = flask.Flask(__name__)
models = {}
models['ISO'] = joblib.load('model.pkl')

@app.route('/models/<model>', methods=['POST'])
def predict(model):
   if (models.get(model) is None):
      return flask.jsonify("[-1]")
   j_data = np.array(flask.request.get_json()['data'])
   y_hat = np.array2string(models[model].predict(j_data))
   return y_hat
if __name__ == '__main__':
   app.run(debug=True,host='0.0.0.0')