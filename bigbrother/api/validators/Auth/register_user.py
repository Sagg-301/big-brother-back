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

        if 'password' not in self.data.keys() or self.data["password"] == "":
            raise ValidationException("The field password field is required")

        if 'confirm_password' not in self.data.keys() or self.data["confirm_password"] == "":
            raise ValidationException("The field confirm_password is required")

        if self.data['confirm_password'] != self.data['password']:
            raise ValidationException("The field confirm_password dont match")

        if 'username' not in self.data.keys() or self.data["username"] == "":
            raise ValidationException("The field username is required")

        if 'email' not in self.data.keys() or self.data["email"] == "":
            raise ValidationException("The field email is required")

        if 'first_name' not in self.data.keys() or self.data["first_name"] == "":
            raise ValidationException("The field first_name is required")

        if 'last_name' not in self.data.keys() or self.data["last_name"] == "":
            raise ValidationException("The field last_name is required")

        if UserModel.objects.filter(username=self.data['username']).first():
            raise ValidationException("That username is already in use")

        if UserModel.objects.filter(email=self.data['email']).first():
            raise ValidationException("That email is already in use")
