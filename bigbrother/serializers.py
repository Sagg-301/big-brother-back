from rest_framework import serializers
from bigbrother.models import *
from django.contrib.auth.models import User as UserModel


class PredictionSerializer(serializers.ModelSerializer):


    class Meta:
        model = Prediction
        fields = ['id', 'x_coordinate', 'y_coordinate',"created_at"]


class UserSerializer(serializers.ModelSerializer):

    class Meta:
        model = UserModel
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'last_login']

class PredictionSerializer(serializers.ModelSerializer):

    class Meta:
        model = Prediction
        fields = ['id', 'x_coordinate', 'y_coordinate', 'created_at']