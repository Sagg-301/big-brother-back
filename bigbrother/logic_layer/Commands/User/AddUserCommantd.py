from ..Command import Command
from bigbrother.data_layer.User import User as UserDAO

class AddUserCommand(Command):
    """
    docstring
    """
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        dao = UserDAO()

        return dao.add(self.payload)