from ..Command import Command
from ...MLModels.mlp import MLPModel

class TrainModelCommand(Command):
    """
    docstring
    """

    def execute(self):
        mlp = MLPModel()
        print("hi")
        mlp.train()