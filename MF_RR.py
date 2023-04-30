from flask import Flask, jsonify, request
from surprise import SVD
from surprise import Dataset,Reader
from surprise import dump
import json
import xgboost as xgb
from xgboost.sklearn import XGBRanker
import pandas as pd
import pickle
import xgboost as xgb

algs, algo = dump.load('Matrix_Factorisation.pkl')
users = pd.read_csv("users_features.csv")
books = pd.read_csv("book_features.csv")
reader = Reader(rating_scale=(1, 10))
train = pd.read_csv("train_df.csv")
data = Dataset.load_from_df(train[['User-ID', 'ISBN', 'Book-Rating']], reader)
trainset = data.build_full_trainset()
app = Flask(__name__)

# model = xgb.XGBRanker()
# model.load_model('xgb.bin')
with open('XGB_Ranker.pkl', 'rb') as file:
    model = pickle.load(file)
    
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
    predictions_sorted = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
    lst =()
    for item, rating in predictions_sorted:
        lst+= ((trainset.to_raw_iid(item),rating),)
    
    dx = pd.DataFrame(lst,columns=["ISBN","Predicted Rating"])
    temp = dx.merge(books[["Book-Title","ISBN"]],on="ISBN",how="inner")
    data_dict = temp.to_dict(orient='records')
    return jsonify(data_dict)



@app.route('/reranked', methods=['POST'])
def reranked():
  user_data = request.get_json()
  user_id = int(user_data["user_id"])
  if user_id not in list(train["User-ID"]):
        return "User Doesn't Exists "
  k=5
  uid = trainset.to_inner_uid(user_id)
  all_items = trainset.all_items()
  x = trainset.ur[int(uid)]
  a = [x[i][0] for i in range(len(x))]
  not_rated_items = [item for item in all_items if item not in a]
  predictions = []
  for iid in not_rated_items:
      pred = algo.predict(uid, trainset.to_raw_iid(iid))
      predictions.append((iid, pred.est))
  predictions_sorted = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
  lst =()
  for item, rating in predictions_sorted:
      lst+= ((trainset.to_raw_iid(item),rating),)
  dx = pd.DataFrame(lst,columns=["ISBN","Predicted Rating"])
  uf =users.loc[users['User-ID'] ==user_id ].drop(columns=['User-ID'])
  temp = dx.merge(books,on="ISBN",how="inner").drop(columns=['page_count',"Predicted Rating","Book-Title"])
  temp["Age"] = users["Age"].iloc[0]
  temp["State"] =users["State"].iloc[0]
  X_test = temp.iloc[:,1:]
  y_pred = model.predict(X_test)
  temp['predicted_score'] = y_pred
  temp = temp.sort_values(by='predicted_score', ascending=False)
  temp = temp[["ISBN"]].merge(books[["Book-Title","ISBN"]],on = 'ISBN',how = 'inner')
  temp = temp[["Book-Title"]]
  temp = temp.iloc[0:5]
  data_dict = temp.to_dict(orient='records')
  return jsonify(data_dict)


 

# Run the Flask app
if __name__ == '__main__':
    app.run(port=8000,debug=True)

