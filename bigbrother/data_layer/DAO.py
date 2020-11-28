from abc import ABC, abstractclassmethod

class DAO(ABC):
    
    @abstractclassmethod
    def add(self, object):
        pass
    
    @abstractclassmethod
    def get(self, id):
        pass

    @abstractclassmethod
    def get_all(self, **kwargs):
        pass

    @abstractclassmethod
    def update(self, **kargs):
        pass

    @abstractclassmethod
    def delete(self, id):
        pass