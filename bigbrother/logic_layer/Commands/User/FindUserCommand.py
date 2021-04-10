from ..Command import Command
from bigbrother.data_layer.User import User as UserDAO
from ....serializers import UserSerializer

class FindUserCommand(Command):
    """
    docstring
    """
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        dao = UserDAO()

        user = dao.find(self.payload)
        user = UserSerializer(user).data

        return user