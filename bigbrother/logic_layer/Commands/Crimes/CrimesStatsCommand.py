from ..Command import Command
from bigbrother.data_layer.Crimes import Crimes as CrimesDAO

class CrimesStatCommand(Command):
    """
    docstring
    """
    def execute(self):
        dao = CrimesDAO()

        stats = dao.stats()

        return stats