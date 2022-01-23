from ..Command import Command
from ...MLModels.lstmrnn import MLPModel

class TrainModelCommand(Command):
    """
    docstring
    """

    def execute(self):
        mlp = MLPModel()
        print("hi")
        mlp.train()