from flask import Flask, jsonify, request
from surprise import SVD
from surprise import Dataset,Reader
from surprise import dump
import json
import pandas as pd
import xgboost as xgb

algs, algo = dump.load('baseline_model.pkl')
reader = Reader(rating_scale=(1, 10))
train = pd.read_csv("train_df.csv")
data = Dataset.load_from_df(train[['User-ID', 'ISBN', 'Book-Rating']], reader)
trainset = data.build_full_trainset()
app = Flask(__name__)


model = xgb.XGBRanker()
model.load_model('xgb.bin')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_data = request.get_json()
    k = 5
    user_id = int(user_data["user_id"])
    if user_id not in list(train["User-ID"]):
        return "User Doesn't Exists "  
    uid = trainset.to_inner_uid(user_id)
    all_items = trainset.all_items()
    x = trainset.ur[int(uid)]
    a = [x[i][0] for i in range(len(x))]
    not_rated_items = [item for item in all_items if item not in a]
    predictions = []
    for iid in not_rated_items:
        pred = algo.predict(uid, trainset.to_raw_iid(iid))
        predictions.append((iid, pred.est))
    predictions_sorted = sorted(predictions, key=lambda x: x[1], reverse=True)[:k]
    lst =()
    for item, rating in predictions_sorted:
        lst+= ((trainset.to_raw_iid(item),rating),)
    
    dict_predictions = dict(lst)
    return jsonify(dict_predictions)

# Run the Flask app
if __name__ == '__main__':
    app.run(port=8000,debug=True)

