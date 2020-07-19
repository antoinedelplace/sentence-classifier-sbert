from flask import Flask
from flask_restful import Resource, Api, reqparse

from sentence_transformers import SentenceTransformer
from joblib import load

model = SentenceTransformer("./model/")
clf = load("./clf.joblib")

app = Flask(__name__)
api = Api(app)

class Prediction(Resource):
    def __init__(self):
        self._required_features = ['input_sentence']
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('input_sentence', type = str, required = True, location = 'json', help = 'No input_sentence provided')
        super(Prediction, self).__init__()

    def post(self):
        args = self.reqparse.parse_args()
        
        embedded_input = model.encode(args['input_sentence'])
        predict = clf.predict(embedded_input)
        
        return {'prediction': int(predict[0])}

api.add_resource(Prediction, '/predict')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')


# For testing purpose :
# curl -i -H "Content-type: application/json" -X POST -d '{"input_sentence": "I am a little confused on all of the models of the 88-89 bonnevilles. I have heard of the LE SE LSE SSE SSEI. Could someone tell me the differences are far as features or performance."}' 127.0.0.1:5000/predict