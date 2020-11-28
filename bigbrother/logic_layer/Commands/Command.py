from abc import ABC, abstractclassmethod

class Command(ABC):
    """ Abstract Command Class """
    
    def __init__(self, data = None):
        self._data = data
        self._response = None

    @abstractclassmethod
    def execute(self):
        """ Execute the command """
        pass

    def response(self):
        """ Return data if excution returned any """
        return self._response