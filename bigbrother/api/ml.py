from django.core.exceptions import ValidationError
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User
from django.utils.translation import gettext as _
import json
import logging
from ..logic_layer.MLModels.mlp import MLPModel

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

        return JsonResponse({'success':1, 'message':"Equis"})
    except Exception as ex:

        logger.exception("Error")

        return JsonResponse({'success': 0, 'error': _('Ha acurrido un error interno')})