
from ....common.Exceptions.ValidationException import ValidationException
from django.contrib.auth.models import User as UserModel

class RegisterUserValidator():
    """
    Validation for Operator Login
    """
    
    def __init__(self, data):
        self.data = data

    def validate(self):
        """
        Validate fields
        """

        if self.data['confirm_password'] != self.data['password']:
            raise ValidationException("La contraseña no coincide")

        if 'username' not in self.data.keys() or self.data["username"] == "":
            raise ValidationException("El campo nombre de usuario es obligatorio")

        if 'email' not in self.data.keys() or self.data["email"] == "":
            raise ValidationException("El campo email es obligatorio")

        if 'first_name' not in self.data.keys() or self.data["first_name"] == "":
            raise ValidationException("El campo nombre es obligatorio")

        if 'last_name' not in self.data.keys() or self.data["last_name"] == "":
            raise ValidationException("El campo apellido es obligatorio")


        if UserModel.objects.get(username = self.data['username']):
            raise ValidationException("Ya existe ese nombre de usuario")

        if UserModel.objects.get(username = self.data['email']):
            raise ValidationException("Ya existe un usuario con ese correo electrónico")
