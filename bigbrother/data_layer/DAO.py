from __future__ import annotations
from abc import ABC, abstractmethod


class DAO(ABC):
    """
    Data Access Object Abstract Class.
    """

    @abstractmethod
    def add(self):
        """ 
        Adds to the Database
        """
        pass

    @abstractmethod
    def get(self):
        """
        Gets alll records from the Database
        """
        pass

    @abstractmethod
    def find(self, id):
        """
        Find a single record from the Database by id
        """
        pass

    @abstractmethod
    def update(self, data):
        """
        Updates a Record on the Database 
        """
        pass

    @abstractmethod
    def delete(self, id):
        """

        Deletes a record on the Database
        """
        pass