# mlp for regression
import numpy as np
import pandas as pd
from django_pandas.io import read_frame

from sklearn import metrics
from pyproj.transformer import Transformer

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from ...data_layer.Crimes import Crimes as CrimesDAO
from ...data_layer.Predictions import Prediction as PredictionDAO
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
        crimes = CrimesDAO().get()
        df = read_frame(crimes)

        return df[:100000]

    def train(self):
        """
        Train the model
        """

        # Part 1 - Data Preprocessing
        # Importing the dataset
        dataset = self.load_data()
        #Irrelevant Data
        dataset = dataset.drop(labels=['case_number','block', 'iucr', 'id','district', 'community_area'], axis=1)

        dataset = pd.get_dummies(dataset, columns=['primary_type'])
        dataset['date'] = pd.to_datetime(dataset["date"])
        dataset.set_index(['date'], inplace=True)

        dataset = dataset.dropna()

        train_dataset = dataset.sample(frac=0.66)
        test_dataset = dataset.drop(train_dataset.index)

        train_dataset['x_coordinate']=(train_dataset['x_coordinate']-train_dataset['x_coordinate'].min())/(train_dataset['x_coordinate'].max()-train_dataset['x_coordinate'].min())
        train_dataset['y_coordinate']=(train_dataset['y_coordinate']-train_dataset['y_coordinate'].min())/(train_dataset['y_coordinate'].max()-train_dataset['y_coordinate'].min())

        train_features = train_dataset.copy()
        test_features = test_dataset.copy()

        print(test_features.columns)

        #Variables dependientes o labels
        train_labels =  pd.concat([train_features.pop('x_coordinate'), train_features.pop('y_coordinate')], axis=1)
        test_labels = pd.concat([test_features.pop('x_coordinate'),test_features.pop('y_coordinate')], axis=1)

        train_features, test_features, train_labels, test_labels = np.array(train_features), np.array(test_features), np.array(train_labels), np.array(test_labels)


        train_features = train_features.reshape(train_features.shape[0],train_features.shape[1],1)
        test_features = test_features.reshape(test_features.shape[0],test_features.shape[1],1)

        
        # normalizer = preprocessing.Normalization()
        # normalizer.adapt(np.array(train_features))

        model = Sequential()
        model.add(LSTM(units = 30, activation = 'relu', input_shape=(train_features.shape[1],1)))
        model.add(Dense(30,activation='relu'))
        model.add(Dense(30,activation='relu'))
        model.add(Dense(2,activation='relu'))

        print(model.summary())

        model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
                        optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

        history = model.fit(
        train_features, train_labels,
        steps_per_epoch = 100,
        # batch_size= 16,
        validation_split=0.1,
        verbose=1, epochs=100, shuffle=False)

        # model.save('models/lstmmodel')

        prediction = model.predict(test_features)
        print(prediction)

        print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, prediction))
        print('Mean Squared Error:', metrics.mean_squared_error(test_labels, prediction))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, prediction)))


    def predict(self, prediction):
        """Predict the location of a crime given the date and the type

        Args:
            prediction ([type]): [description]

        Returns:
            [type]: [description]
        """

        #Get the max and min coordinates for conversion purposes

        #Transformar de NAD83 a coordenadas universales
        transformer = Transformer.from_crs( "epsg:3602","epsg:4326",always_xy=False)
        min_max = CrimesDAO().get_min_max_coordinates()

        model = tf.keras.models.load_model('models/lstmmodel')
        data = pd.DataFrame(prediction['data'], index=[0])
        data.set_index(['date'], inplace=True)

        data = np.array(data)

        data = data.reshape(data.shape[0],data.shape[1],1)

        response = model.predict(data)

        response = response[0]
        print(response)
        response[0] = int(response[0] * (min_max['max_x'] - min_max['min_x']) + min_max['min_x'])
        response[1] = int(response[1] * (min_max['max_y'] - min_max['min_y']) + min_max['min_y'])

        predictionDAO = PredictionDAO()
        predictionDAO.add({'x_coordinate':response[0],'y_coordinate':response[1], 'user_id': prediction['user_id']})

        #Transformar a coordenadas lat lon
        response[0], response[1] = transformer.transform(response[0] / 3.28, response[1] / 3.28)
        return response
       
