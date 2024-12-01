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
from itertools import product
import random

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
        return np.argsort(y_pred, axis=1)[:, -4:]

    def score(self, X, y):
        # 평가
        y_one_hot = to_categorical(y, num_classes=self.class_size)
        loss, accuracy = self.model.evaluate(X, y_one_hot)
        return accuracy

##########
# 모델 경로 수정
start_pipeline = joblib.load('/content/drive/MyDrive/busan/start_exercise_predict_model.pkl')
main_pipeline = joblib.load('/content/drive/MyDrive/busan/main_exercise_predict_model.pkl')
end_pipeline = joblib.load('/content/drive/MyDrive/busan/end_exercise_predict_model.pkl')

# 파라미터로 넘겨 받을 것
age = 52
sex = 1
height = 180.8
weight = 80.8
fat = 17.6
blood_pressure_low = 80.0
blood_pressure_high = 140.0

start_exercise = start_pipeline.predict([[age, sex, height, weight, fat, blood_pressure_low, blood_pressure_high]])
main_exercise = main_pipeline.predict([[age, sex, height, weight, fat, blood_pressure_low, blood_pressure_high]])
end_exercise = end_pipeline.predict([[age, sex, height, weight, fat, blood_pressure_low, blood_pressure_high]])

start_exercise = [item for sublist in start_exercise for item in sublist]
main_exercise = [item for sublist in main_exercise for item in sublist]
end_exercise = [item for sublist in end_exercise for item in sublist]

combinations = list(product(start_exercise, main_exercise, end_exercise))

res = []

for i in range(len(combinations)):
    if combinations[i][0] != combinations[i][1] and combinations[i][0] != combinations[i][2] and combinations[i][1] != combinations[i][2]:
        res.append(combinations[i])

# 최대 64개 조합중 7만 추천됨
return_value = random.sample(res, 7)
return_value