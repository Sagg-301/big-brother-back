from django.core.exceptions import ValidationError
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User
from django.utils.translation import gettext as _
import json
import logging
from ..logic_layer.MLModels.lstmrnn import MLPModel
from ..logic_layer.Commands import *

logger = logging.getLogger(__name__)

@csrf_exempt
def train_mlp(request):
    """ Api endpoint to register user"""
    try:
        # body = json.loads(request.body.decode('UTF-8'), encoding='UTF-8')
        # command = RegisterUserCommand(body)
        # command.execute()

        mlm = MLPModel()
        mlm.train()

        return JsonResponse({'success':1, 'message':"Trained Successfully"})
    except Exception as ex:

        logger.exception("Error")

        return JsonResponse({'success': 0, 'error': _('Ha acurrido un error interno')})

@csrf_exempt
def predict(request):
    """ Api endpoint to register user"""
    try:
        body = json.loads(request.body.decode('UTF-8'), encoding='UTF-8')
        command = PredictLocationCommand(body)
        response = command.execute().tolist()

        return JsonResponse({'success':1, 'message':"Predicción realizada con éxito", 'data':{
            'x_coordinate':response[0],
            'y_coordinate':response[1],
        }})
    except Exception as ex:

        logger.exception("Error")

        return JsonResponse({'success': 0, 'error': _('Ha acurrido un error interno')})