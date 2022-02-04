from .DAO import DAO
from django.contrib.auth.models import User as UserModel

class User(DAO):
    
    def __init__(self):
        pass
    
    def add(self, data):
        try:
            user = UserModel.objects.create_user(data['username'], data['email'], data['password'])
            user.first_name = data['first_name']
            user.last_name = data['last_name']
            user.is_staff = True

            user.save()

            return user
        except Exception as ex:
            raise ex
    
    def get(self):
        try:
            users = UserModel.objects.all().filter(is_superuser = False)

            return users
        except Exception as ex:
            raise ex

    
    def find(self, id):
        try:
            user = UserModel.objects.get(pk=id)

            return user
        except Exception as ex:
            raise ex

    
    def update(self, data):
        try:
            user = UserModel.objects.get(pk=data['id'])

            user.first_name = data['first_name']
            user.last_name = data['last_name']
            user.username = data['username']
            user.email = data['email']
            if data['password']:
                user.password = data['password']

            user.save()

            return user.id
        except Exception as ex:
            raise ex
    
    def delete(self, id):
        try:
            user = UserModel.objects.get(pk=id)

            user.is_active = not user.is_active

            user.save()

            return user.is_active
        except Exception as ex:
            raise ex