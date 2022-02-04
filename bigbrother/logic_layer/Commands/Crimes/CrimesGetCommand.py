from ..Command import Command
from bigbrother.data_layer.Crimes import Crimes as CrimesDAO
from ....serializers import CrimeSerializer

class CrimesGetCommand(Command):
    """
    docstring
    """
    def execute(self):
        dao = CrimesDAO()

        crimes = dao.get()

        return CrimeSerializer(crimes, many=True).data