# mlp for regression
import numpy as np
import pandas as pd
from django_pandas.io import read_frame
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from ...data_layer.Crimes import Crimes as CrimesDAO
import joblib
from sklearn import metrics
from pyproj.transformer import Transformer
from ...data_layer.Predictions import Prediction as PredictionDAO

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

        return df

    def train(self):
        """
        Train the model
        """

        # Parte 1 - Preoprocesamiento de datos
        # importando datos
        dataset = self.load_data()
        #Irrelevant Data
        dataset = dataset.drop(labels=['case_number','block', 'iucr', 'id','district', 'community_area'], axis=1)

        #Cambiar fecha a timestamp para hacerlo un entero continuo
        dataset = pd.get_dummies(dataset, columns=['primary_type'])
        dataset['date'] = pd.to_datetime(dataset["date"])
        dataset.set_index(['date'], inplace=True)

        #Descartar filas con datos vacios
        dataset = dataset.dropna()

        #Se divide el conjunto de datos
        train_dataset = dataset.sample(frac=0.66)
        test_dataset = dataset.drop(train_dataset.index)

        train_features = train_dataset.copy()
        test_features = test_dataset.copy()

        #Variables dependientes o labels
        train_labels =  pd.concat([train_features.pop('x_coordinate'), train_features.pop('y_coordinate')], axis=1)
        test_labels = pd.concat([test_features.pop('x_coordinate'),test_features.pop('y_coordinate')], axis=1)

        train_features, test_features, train_labels, test_labels = np.array(train_features), np.array(test_features), np.array(train_labels), np.array(test_labels)

        model = RandomForestRegressor(n_estimators=1000, random_state=42, verbose=1)
        model.fit(train_features, train_labels)
       
        prediction = model.predict(test_features)
        

        print(prediction)
        print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, prediction))
        print('Mean Squared Error:', metrics.mean_squared_error(test_labels, prediction))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, prediction)))

        joblib.dump(model, 'models/rfmodel')

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

        model = joblib.load('models/rfmodel')
        data = pd.DataFrame(prediction['data'], index=[0])
        data.set_index(['date'], inplace=True)
        data = np.array(data)

        print(data.shape)

        response = model.predict(data)

        response = response[0]
        print(response)

        predictionDAO = PredictionDAO()
        predictionDAO.add({'x_coordinate':response[0],'y_coordinate':response[1], 'user_id': prediction['user_id']})

        #Transformar a coordenadas lat lon
        response[0], response[1] = transformer.transform(response[0] / 3.28, response[1] / 3.28)
        return response