from __future__ import annotations
from abc import ABC, abstractmethod
from django.db import connection


class DAO(ABC):
    """
    Data Access Object Abstract Class.
    """
    def raw_query(self, query:str = "", data:list = []):
        """Execute a raw query

        Args:
            query (str, optional): Query to execute. Defaults to "".
            data (list, optional): Variables. Defaults to [].

        Returns:
            [type]: [description]
        """
        with connection.cursor() as cursor:
            cursor.execute(query, data)
            rows = cursor.fetchall()

        return rows

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