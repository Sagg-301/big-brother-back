# mlp for regression
import numpy as np
import pandas as pd
from pandas.core.arrays import categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from django_pandas.io import read_frame
from ...models import CrimesData
from sklearn.compose import ColumnTransformer 
import tensorflow as tf
import datetime as dt

# import sys
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

        #Transform date to integer
        dataset['date'] = pd.to_datetime(dataset['date'])
        dataset['date']=dataset['date'].map(dt.datetime.toordinal)  

        X = dataset[['date','primary_type']].values
        y = dataset['district'].values

        # Encoding categorical data
        labelencoder_X = LabelEncoder()
        X[:,1] = labelencoder_X.fit_transform(X[:,1])

        ct = ColumnTransformer([("onehot", OneHotEncoder(handle_unknown='ignore'), [1])], remainder="passthrough") # The last arg ([0]) is the list of columns you want to transform in this step
        X = ct.fit_transform(X).toarray()

        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Part 2 - Now let's make the ANN!

        # Importing the Keras libraries and packages

        # Initialising the ANN
        model = Sequential()

        # Adding the input layer and the first hidden layer
        model.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu', input_dim = 27))

        # Adding the second hidden layer
        model.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))

        # Adding the third hidden layer
        model.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))

        # Adding the output layer
        model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

        model.summary()

        # Compiling the ANN
        model.compile(optimizer = tf.optimizers.Adam(learning_rate=0.1), loss = 'mean_absolute_error', metrics = ['accuracy'])

        # Fitting the ANN to the Training set
        model.fit(X_train, y_train, batch_size = 10, epochs = 100, verbose= 0)

        model.save('trained-model')

        # Part 3 - Making the predictions and evaluating the model

        # Predicting the Test set results
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5)
        print(y_pred)

        # Making the Confusion Matrix
        # cm = confusion_matrix(y_test, y_pred)
        # print(cm)

    def predict(parameter_list):
        """
        docstring
        """
        pass