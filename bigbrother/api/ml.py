from django.http import JsonResponse
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework import viewsets
from django.utils.translation import gettext as _
import json
import logging
from ..logic_layer.MLModels.random_forest import MLPModel
from ..logic_layer.Commands import *

logger = logging.getLogger(__name__)

class MLApiView(viewsets.ViewSet):

    # permission_classes = (IsAuthenticated,)

    @action(methods=['post'], detail=False, permission_classes=[IsAuthenticated],
            url_path='train', url_name='train')
    def train(self, request):
        """ Api endpoint to register user"""
        try:

            mlm = MLPModel()
            mlm.train()

            return JsonResponse({'success':1, 'message':"Trained Successfully"})
        except Exception as ex:

            logger.exception("Error")

            return JsonResponse({'success': 0, 'error': _('There is an error at retrain IA')},500)

    @action(methods=['post'], detail=False, permission_classes=[IsAuthenticated],
            url_path='predict', url_name='predict')
    def predict(self, request):
        """ Api endpoint to register user"""
        try:
            body = json.loads(request.body.decode('UTF-8'), encoding='UTF-8')

            command = PredictLocationCommand({"data":body,"user_id": request.user.id})
            response = command.execute().tolist()

            return JsonResponse({'success':1, 'message':"Predicción realizada con éxito", 'data':{
                'x_coordinate':response[0],
                'y_coordinate':response[1],
            }})
        except Exception as ex:

            logger.exception("Error")

            return JsonResponse({'success': 0, 'error': _('Ha acurrido un error interno')})