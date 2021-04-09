from abc import ABC, abstractclassmethod

class Command(ABC):
    """ Abstract Command Class """

    @abstractclassmethod
    def execute(self):
        """ Execute the command """
        pass