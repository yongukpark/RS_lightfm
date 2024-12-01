import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
import joblib

def make_datas(data, target): # target = 'start' or 'main' or 'end'
  return data.iloc[:, :7], data[target]

def cal_class_weights(target):
  target_class_size = 687 # 타깃 레이블 개수
  class_weights = {}
  for i in range(target_class_size):
    dic = 1
    a = np.sum(target == i)
    if a > 3000:
      a = a/3000
      dic = dic/a
    class_weights[i] = dic
  return class_weights


class ExerciseClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=50, batch_size=64, class_size=687, class_weights=[]):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.class_size = class_size
        self.class_weights = class_weights
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    def create_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=7, activation='relu'))
        model.add(BatchNormalization())  # 배치 정규화 추가
        model.add(Dropout(0.3))  # Dropout 추가
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))  # Dropout 추가
        model.add(Dense(self.class_size, activation='softmax'))

        # 옵티마이저 설정
        optimizer = Adam(learning_rate=0.001)

        # 모델 컴파일
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X, y):
        self.model = self.create_model()

        # 원-핫 인코딩
        y_one_hot = to_categorical(y, num_classes=self.class_size)

        # 모델 학습
        self.model.fit(X, y_one_hot, epochs=self.epochs, batch_size=self.batch_size,
                       class_weight=self.class_weights, validation_split=0.2, callbacks=[self.early_stopping], verbose=1)
        return self

    def predict(self, X):
        # 예측
        y_pred = self.model.predict(X)
        return np.argsort(y_pred, axis=1)[:, -3:]

    def score(self, X, y):
        # 평가
        y_one_hot = to_categorical(y, num_classes=self.class_size)
        loss, accuracy = self.model.evaluate(X, y_one_hot)
        return accuracy

def solution(data, y):
  X,y = make_datas(data, 'start')
  class_weights = cal_class_weights(y)

  model = ExerciseClassifier(epochs=50, batch_size=64, class_weights=class_weights)
  pipeline = Pipeline([
      ('scaler', StandardScaler()),  # 데이터 스케일링
      ('model', model)  # Keras 모델
  ])

  pipeline.fit(X, y)
  return pipeline

################
a = 'main' # 여기만 받아오면 됨 start,main,end
data = pd.read_csv('/content/drive/MyDrive/busan/output(random_duplication).csv') # 경로 수정 해야함
pipeline = solution(data, a)
joblib.dump(pipeline, a+'_exercise_predict_model.pkl')