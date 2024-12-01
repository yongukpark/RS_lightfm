# numpy==1.23.0 이하 버전 필수

# !pip install lightfm
# !pip install lightfm-dataset-helper
# 깔려 있어야 동작함

import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm_dataset_helper.lightfm_dataset_helper import DatasetHelper
from scipy.sparse import csr_matrix
import joblib

def call_data(target):
  user_features = pd.read_csv('/content/drive/MyDrive/busan/user_features.csv')
  score = pd.read_csv('/content/drive/MyDrive/busan/'+target+'_score.csv')
  return user_features, score

def make_dataset(features, score, target):
  exercise = pd.DataFrame(np.arange(687))
  exercise.columns = [target]
  score_size = len(score)
  exercise["weight"] = 1.0

  items_feature_columns = ["weight"]
  items_column = target
  user_column = "idx"
  ratings_column = "score"
  user_features_columns = ['MESURE_AGE_CO',	'SEXDSTN_FLAG_CD',	'MESURE_IEM_001_VALUE',	'MESURE_IEM_002_VALUE',	'MESURE_IEM_003_VALUE',	'MESURE_IEM_005_VALUE',	'MESURE_IEM_006_VALUE']
  ratings = score 
  
  dataset_helper_instance = DatasetHelper(
    users_dataframe=features,
    items_dataframe=exercise,
    interactions_dataframe=ratings,
    item_id_column=items_column,
    items_feature_columns=items_feature_columns,
    user_id_column=user_column,
    user_features_columns=user_features_columns,
    interaction_column=ratings_column,
    clean_unknown_interactions=True,
  )
  return dataset_helper_instance

  
def train_model(dataset):
  model = LightFM(no_components=24, loss="warp", k=15)
  model.fit(
      interactions=dataset.interactions,
      sample_weight=dataset.weights,
      item_features=dataset.item_features_list,
      user_features=dataset.user_features_list,
      verbose=True,
      epochs=10,
      num_threads=20,
  )
  return model


def solution(target):
  features, score = call_data(target)
  dataset = make_dataset(features, score, target)
  dataset.routine()
  model = train_model(dataset)
  return model, dataset
  
##########
# input은 따로 없고 코드 실행시
# 신규유입자 + 신규 선호도? 를 기준으로 재학습 시킴
# 이거를 업데이트 주기 조정해서 주기적으로 업데이트를 해야하는지 
# 신규유저가 들어왔을 때 업데이트 행야하는지 생각해야함

# 현재 학습시간 1분정도
li = ['start','main','end']
for a in li:
  model, dataset = solution(a)

  joblib.dump(model, a+'_recommend_model.pkl')
  joblib.dump(dataset, a+'_lightfm_dataset.pkl')