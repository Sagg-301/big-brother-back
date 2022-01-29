from ..Command import Command
from bigbrother.data_layer.User import User as UserDAO

class UpdateUserCommand(Command):
    """
    docstring
    """
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        dao = UserDAO()

        return dao.update(self.payload)