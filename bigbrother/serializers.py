from rest_framework import serializers
from bigbrother.models import *
from django.contrib.auth.models import User as UserModel


class PredictionSerializer(serializers.ModelSerializer):


    class Meta:
        model = Prediction
        fields = ['id', 'x_coordinate', 'y_coordinate',"created_at"]

class CrimeSerializer(serializers.ModelSerializer):


    class Meta:
        model = CrimesData
        fields = ['case_number','x_coordinate', 'y_coordinate',"date", "district", "primary_type"]


class UserSerializer(serializers.ModelSerializer):

    class Meta:
        model = UserModel
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'last_login', 'is_active', 'is_superuser']

class PredictionSerializer(serializers.ModelSerializer):

    class Meta:
        model = Prediction
        fields = ['id', 'x_coordinate', 'y_coordinate', 'created_at']