# 얘네 설치해야함
# !pip install lightfm
# !pip install lightfm-dataset-helper

start_model = joblib.load('/content/drive/MyDrive/busan/start_recommend_model.pkl')
main_model = joblib.load('/content/drive/MyDrive/busan/main_recommend_model.pkl')
end_model = joblib.load('/content/drive/MyDrive/busan/end_recommend_model.pkl')

start_dataset = joblib.load('/content/drive/MyDrive/busan/start_lightfm_dataset.pkl')
main_dataset = joblib.load('/content/drive/MyDrive/busan/main_lightfm_dataset.pkl')
end_dataset = joblib.load('/content/drive/MyDrive/busan/end_lightfm_dataset.pkl')

import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm_dataset_helper.lightfm_dataset_helper import DatasetHelper
import joblib
import random
from itertools import product

def recommand_real(scores):

  real_scores = scores
  real_items = np.argsort(real_scores)[::-1][:3]
  return real_items

def recommand_random(scores):
  # random recommend

  random_scores = scores + np.random.uniform(1,20,len(scores))
  random_items = np.argsort(random_scores)[::-1][:5]
  return random_items

def predict(model, dataset, id):

  scores = model.predict(user_ids=id,
      item_ids=np.arange(687),
      item_features=dataset.item_features_list,
      user_features=dataset.user_features_list)

  real_items = recommand_real(scores)
  random_items = recommand_random(scores)

  return_list = []
  for i in real_items:
    return_list.append(i)
  for i in random_items:
    return_list.append(i)

  return_list = list(set(return_list))
  random.shuffle(return_list)
  return return_list

##########
# 입력 : id
id = 1000
start = predict(start_model, start_dataset, id-1)
main = predict(main_model, main_dataset, id-1)
end = predict(end_model, end_dataset, id-1)
res = []
combinations = list(product(start, main, end))
for i in range(len(combinations)):
    if combinations[i][0] != combinations[i][1] and combinations[i][0] != combinations[i][2] and combinations[i][1] != combinations[i][2]:
        res.append(combinations[i])
res = random.sample(res, 4)
# 출력 : 사용자의 특성과 선호도를 고려해서 총 4쌍을 추천
print(res)