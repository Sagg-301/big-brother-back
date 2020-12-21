# mlp for regression
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers.experimental import preprocessing
from django_pandas.io import read_frame
from ...models import CrimesData
import datetime as dt

import sys
np.set_printoptions(threshold=sys.maxsize)
# load the dataset

class MLPModel(object):
    """
    Multi Layer Perceptron Model
    """
    def load_data(self):
        """
        Load the data
        """
        crimes = CrimesData.objects.all()
        df = read_frame(crimes)

        return df

    def train(self):
        """
        Train the model
        """

        # Part 1 - Data Preprocessing
        # Importing the dataset
        dataset = self.load_data()
        dataset = dataset.drop(labels=['case_number','block', 'iucr', 'id'], axis=1)\

        #Transform date to integer
        dataset['date'] = pd.to_datetime(dataset['date'])
        dataset['date']=dataset['date'].map(dt.datetime.toordinal)

        dataset = pd.get_dummies(dataset,columns=['primary_type', 'district'])

        dataset = dataset.dropna()

        train_dataset = dataset.sample(frac=0.66, random_state= 1)
        test_dataset = dataset.drop(train_dataset.index)

        #Variables independientes o features
        train_features = train_dataset.copy()
        test_features = test_dataset.copy()

        #Variables dependientes o labels
        train_labels = train_features.pop('date')
        test_labels = test_features.pop('date')

        FEATURES = 60
        N_VALIDATION = int(1e3)
        N_TRAIN = int(1e4)
        BUFFER_SIZE = int(1e4)
        BATCH_SIZE = 500
        STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

        #Normalizar
        normalizer = preprocessing.Normalization()
        normalizer.adapt(np.array(train_features))

        model = keras.Sequential([
            normalizer,
            layers.Dense(512,activation='relu', input_shape=(FEATURES,)),
            layers.Dense(512,activation='relu'),
            layers.Dense(512,activation='relu'),
            layers.Dense(512,activation='relu'),
            layers.Dense(1)
        ])
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001,
            decay_steps=STEPS_PER_EPOCH*1000,
            decay_rate=1,
            staircase=False)

        model.compile(loss='mean_absolute_error',
                        optimizer=tf.keras.optimizers.Adam(lr_schedule))
        
        print(model.summary())
        history = model.fit(
        train_features, train_labels,
        steps_per_epoch = STEPS_PER_EPOCH,
        validation_split=0.3,
        verbose=1, epochs=1000)

        test_results = model.evaluate(test_features, test_labels, verbose=0)
        print(test_results)

        # Making the Confusion Matrix
        # cm = confusion_matrix(y_test, y_pred)
        # print(cm)

    def predict(parameter_list):
        """
        docstring
        """
        pass