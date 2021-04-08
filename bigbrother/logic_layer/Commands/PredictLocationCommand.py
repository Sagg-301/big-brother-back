from .Command import Command
from ..MLModels.lstmrnn import MLPModel

class PredictLocationCommand(Command):
    """
    docstring
    """
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        mlp = MLPModel()

        mlp.predict(self.payload)