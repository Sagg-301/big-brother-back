from .DAO import DAO
from ..models import Prediction as PredictionModel
from pyproj import Transformer 

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

        #Transformar de NAD83 a coordenadas universales
        transformer = Transformer.from_crs( "epsg:3602","epsg:4326",always_xy=False)
        predictions = PredictionModel.objects.all()
        print(predictions)
        for pr in predictions:
            pr.x_coordinate, pr.y_coordinate = transformer.transform(pr.x_coordinate / 3.28, pr.y_coordinate / 3.28)
        return predictions

    def find(self, id):
        pass
    
    def update(self, data):
        pass
    
    def delete(self, id):
        pass