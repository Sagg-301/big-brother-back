from ..Command import Command
from bigbrother.data_layer.Predictions import Prediction as PredictionDAO
from ....serializers import PredictionSerializer

class GetPredictionsCommand(Command):
    """
    docstring
    """
    def execute(self):
        dao = PredictionDAO()

        predictions = dao.get()
        predictions = PredictionSerializer(predictions, many=True).data

        return predictions