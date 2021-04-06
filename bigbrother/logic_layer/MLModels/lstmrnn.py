# mlp for regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from django_pandas.io import read_frame
from numpy import nan

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers.experimental import preprocessing

from sklearn.metrics import mean_squared_error
from ...models import CrimesData
import tensorflow as tf

import sys
# np.set_printoptions(threshold=sys.maxsize)
# load the dataset

class MLPModel(object):
    """
    Multi Layer Perceptron Model
    """
    def load_data(self):
        """
        Load the data
        """
        crimes = CrimesData.objects.all()[:10000]
        df = read_frame(crimes)

        return df

    def train(self):
        """
        Train the model
        """

        # Part 1 - Data Preprocessing
        # Importing the dataset
        dataset = self.load_data()
        #Irrelevant Data
        dataset = dataset.drop(labels=['case_number','block', 'iucr', 'id','x_coordinate','y_coordinate', 'community_area'], axis=1)

        dataset = pd.get_dummies(dataset, columns=['primary_type'])
        dataset['date'] = pd.to_datetime(dataset["date"])
        dataset.set_index(['date'], inplace=True)

        dataset = dataset.dropna()


        train_dataset = dataset.sample(frac=0.66)
        test_dataset = dataset.drop(train_dataset.index)

        train_features = train_dataset.copy()
        test_features = test_dataset.copy()

        #Variables dependientes o labels
        train_labels = pd.get_dummies(train_features.pop('district'), columns="district")
        test_labels = pd.get_dummies(test_features.pop('district'), columns="district")

        train_features, test_features, train_labels, test_labels = np.array(train_features), np.array(test_features), np.array(train_labels), np.array(test_labels)


        train_features = train_features.reshape(train_features.shape[0],train_features.shape[1],1)
        test_features = test_features.reshape(test_features.shape[0],test_features.shape[1],1)
        # normalizer = preprocessing.Normalization()
        # normalizer.adapt(np.array(train_features))

        model = Sequential()
        model.add(LSTM(units = 512, activation = 'relu', input_shape=(train_features.shape[1],1)))
        model.add(Dense(512,activation='relu'))
        model.add(Dense(512,activation='relu'))
        model.add(Dense(512,activation='relu'))
        model.add(Dense(train_labels.shape[1],activation='softmax'))

        print(model.summary())

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                        optimizer=tf.keras.optimizers.Adam(0.005), metrics=['accuracy'])

        history = model.fit(
        train_features, train_labels,
        # steps_per_epoch = 100,
        # batch_size= 16,
        validation_split=0.1,
        verbose=1, epochs=100, shuffle=False)

        model.save('models/lstmmodel')

    # def predict(self, prediction):
    #     """
    #     docstring
    #     """
    #     model = tf.keras.models.load_model('models/model')
    #     data = pd.DataFrame(prediction, index=[0])
    #     print(model.summary())
    #     response = model.predict(data)

    #     print(response)
