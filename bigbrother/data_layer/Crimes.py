from .DAO import DAO
from ..models import CrimesData

class Crimes(DAO):
    
    def __init__(self):
        pass
    
    def add(self, data):
        pass
    
    def get(self):
        crimes = CrimesData.objects.all()
        return crimes

    def find(self, id):
        pass
    
    def update(self, data):
        pass
    
    def delete(self, id):
        pass

    def get_min_max_coordinates(self):
        coordinates = self.raw_query(""" SELECT (SELECT "X Coordinate" FROM crimes_data WHERE "X Coordinate" IS NOT NULL ORDER BY "X Coordinate" DESC LIMIT 1) AS max_x,
                                            (SELECT "X Coordinate" FROM crimes_data WHERE "X Coordinate" IS NOT NULL ORDER BY "X Coordinate" ASC LIMIT 1) AS min_x,
                                            (SELECT "Y Coordinate" FROM crimes_data WHERE "Y Coordinate" IS NOT NULL ORDER BY "Y Coordinate" DESC LIMIT 1) AS max_y,
                                            (SELECT "Y Coordinate" FROM crimes_data WHERE "Y Coordinate" IS NOT NULL ORDER BY "Y Coordinate" ASC LIMIT 1) AS min_y""")[0]
                                

        return coordinates