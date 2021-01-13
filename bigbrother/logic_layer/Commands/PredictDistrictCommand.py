from .Command import Command
from ..MLModels.mlp import MLPModel

class PredictDistrictCommand(Command):
    """
    docstring
    """
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        mlp = MLPModel()

        mlp.predict(self.payload)