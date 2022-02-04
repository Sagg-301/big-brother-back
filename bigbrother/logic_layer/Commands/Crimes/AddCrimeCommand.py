from ..Command import Command
from bigbrother.data_layer.Crimes import Crimes as CrimesDAO

class AddCrimeCommand(Command):

    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        dao = CrimesDAO()

        stats = dao.add(self.payload)

        return stats