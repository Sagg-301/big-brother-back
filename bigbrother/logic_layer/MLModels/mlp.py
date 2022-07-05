# mlp for regression
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers.experimental import preprocessing
from django_pandas.io import read_frame
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras import utils
from ...models import CrimesData
from sklearn import metrics

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
        crimes = CrimesData.objects.all()
        df = read_frame(crimes)

        return df[:100000]

    def dataframe_to_dataset(dataframe):
        dataframe = dataframe.copy()
        labels = dataframe.pop("target")
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        ds = ds.shuffle(buffer_size=len(dataframe))
        return ds

    def train(self):
        """
        Train the model
        """

        # Part 1 - Data Preprocessing
        # Importing the dataset
        dataset = self.load_data()
        dataset = dataset.drop(labels=['case_number','block', 'iucr', 'id','district', 'community_area'], axis=1)

        dataset = pd.get_dummies(dataset, columns=['primary_type'])
        dataset['date'] = pd.to_datetime(dataset["date"])
        dataset['date'] = (dataset['date'] - dataset['date'].min())/(dataset['date'].max()-dataset['date'].min())
        # dataset.set_index(['date'], inplace=True)

        dataset = dataset.dropna()

        train_dataset = dataset.sample(frac=0.66)
        test_dataset = dataset.drop(train_dataset.index)

        train_dataset['x_coordinate']=(train_dataset['x_coordinate']-train_dataset['x_coordinate'].min())/(train_dataset['x_coordinate'].max()-train_dataset['x_coordinate'].min())
        train_dataset['y_coordinate']=(train_dataset['y_coordinate']-train_dataset['y_coordinate'].min())/(train_dataset['y_coordinate'].max()-train_dataset['y_coordinate'].min())

        train_features = train_dataset.copy()
        test_features = test_dataset.copy()

        #Variables dependientes o labels
        train_labels =  pd.concat([train_features.pop('x_coordinate'), train_features.pop('y_coordinate')], axis=1)
        test_labels = pd.concat([test_features.pop('x_coordinate'),test_features.pop('y_coordinate')], axis=1)

        print(test_features.columns)

        FEATURES = len(train_features.columns)
        LABELS = len(train_labels.columns)

        # Normalizar
        normalizer = preprocessing.Normalization()
        normalizer.adapt(np.array(train_features))
        print(train_features)

        model = keras.Sequential([
            # normalizer,
            layers.Dense(FEATURES ,activation='tanh', input_shape=(FEATURES,), kernel_regularizer='l1'),
            layers.Dense(128,activation='tanh'),
            layers.Dense(100,activation='tanh'),
            layers.Dense(100,activation='tanh'),
            layers.Dense(50,activation='relu', kernel_regularizer='l1'),
            Dropout(0.3),
            layers.Dense(LABELS,activation='softmax')
        ])

        model.compile(loss='mean_squared_error',
                        optimizer=tf.keras.optimizers.Adam(0.0005), metrics=['accuracy'])
        
        # tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

        history = model.fit(
        train_features, train_labels,
        steps_per_epoch = 1000,
        validation_split=0.3,
        verbose=1, epochs=100)

        test_results = model.evaluate(test_features, test_labels)
        print(test_results)

        model.save('models/dnnmodel')

        prediction = model.predict(test_features)
        print(prediction)

        print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, prediction))
        print('Mean Squared Error:', metrics.mean_squared_error(test_labels, prediction))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, prediction)))

    def predict(self, prediction):
        """
        docstring
        """
        model = tf.keras.models.load_model('models/model')
        data = pd.DataFrame(prediction, index=[0])
        print(model.summary())
        response = model.predict(data)

        print(response)
