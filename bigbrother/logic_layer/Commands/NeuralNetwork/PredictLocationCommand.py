from ..Command import Command
from ...MLModels.random_forest import MLPModel

class PredictLocationCommand(Command):
    """
    docstring
    """
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        mlp = MLPModel()

        return mlp.predict(self.payload)