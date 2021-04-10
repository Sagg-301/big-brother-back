from rest_framework import serializers
from bigbrother.models import *


class PredictionSerializer(serializers.ModelSerializer):

    class Meta:
        model = Prediction
        fields = ('id', 'x_coordinate', 'y_coordinate')