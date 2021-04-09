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
import datetime as dt

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
        crimes = CrimesData.objects.all()[:100000]
        df = read_frame(crimes)

        return df

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
        dataset = dataset.drop(labels=['case_number','block', 'iucr', 'id','x_coordinate','y_coordinate', 'community_area'], axis=1)

        #Transform date to integer
        dataset['date'] = pd.to_datetime(dataset['date'])
        dataset['date']=dataset['date'].map(dt.datetime.toordinal)

        # dataset['date']=(dataset['date']-dataset['date'].min())/(dataset['date'].max()-dataset['date'].min())

        dataset = pd.get_dummies(dataset, columns=['primary_type'])
        # dataset['primary_type'] = pd.Categorical(dataset['primary_type'])
        # dataset['primary_type'] = dataset['primary_type'].cat.codes

        print(dataset)

        dataset = dataset.dropna()

        train_dataset = dataset.sample(frac=0.66, random_state= 1)
        test_dataset = dataset.drop(train_dataset.index)

        #Variables independientes o features
        train_features = train_dataset.copy()
        test_features = test_dataset.copy()

        #Variables dependientes o labels
        train_labels = pd.get_dummies(train_features.pop('district'),columns=['district'])
        test_labels = pd.get_dummies(test_features.pop('district'),columns=['district'])

        print(test_features.columns)

        FEATURES = len(train_features.columns)
        LABELS = len(train_labels.columns)

        # Normalizar
        normalizer = preprocessing.Normalization()
        normalizer.adapt(np.array(train_features))

        model = keras.Sequential([
            # normalizer,
            layers.Dense(FEATURES ,activation='relu', input_shape=(FEATURES,)),
            layers.Dense(30,activation='relu'),
            layers.Dense(512,activation='relu'),
            layers.Dense(30,activation='relu'),
            Dropout(0.1),
            layers.Dense(LABELS,activation='softmax')
        ])

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                        optimizer=tf.keras.optimizers.Adam(0.0005), metrics=['accuracy'])
        
        # tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

        history = model.fit(
        train_features, train_labels,
        steps_per_epoch = 500,
        validation_split=0.3,
        verbose=1, epochs=1000)

        test_results = model.evaluate(test_features, test_labels)
        print(test_results)

        model.save('models/model')

        # result = model.predict(test_features)
        # print(result)

        # Making the Confusion Matrix
        # cm = confusion_matrix(y_test, y_pred)
        # print(cm)

    def predict(self, prediction):
        """
        docstring
        """
        model = tf.keras.models.load_model('models/model')
        data = pd.DataFrame(prediction, index=[0])
        print(model.summary())
        response = model.predict(data)

        print(response)
