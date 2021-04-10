from ..Command import Command
from bigbrother.data_layer.User import User as UserDAO
from ....serializers import UserSerializer

class GetUsersCommand(Command):
    """
    docstring
    """
    def execute(self):
        dao = UserDAO()

        users = dao.get()
        users = UserSerializer(users, many=True).data

        return users