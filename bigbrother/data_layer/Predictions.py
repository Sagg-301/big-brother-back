from .DAO import DAO
from ..models import Prediction as PredictionModel

class Prediction(DAO):
    
    def __init__(self):
        pass
    
    def add(self, data):
        try:
            prediction = PredictionModel()
            prediction.x_coordinate = data['x_coordinate']
            prediction.y_coordinate = data['y_coordinate']
            prediction.user_id = data['user_id']

            prediction.save()

            return prediction
        except Exception as ex:
            raise ex
    
    def get(self):
        predictions = PredictionModel.objects.all()

        return predictions

    def find(self, id):
        pass
    
    def update(self, data):
        pass
    
    def delete(self, id):
        pass