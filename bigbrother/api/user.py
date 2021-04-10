from django.core.exceptions import ValidationError
from django.http import JsonResponse
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework import viewsets
from rest_framework.response import Response
from django.utils.translation import gettext as _
import json
import logging
from ..logic_layer.MLModels.lstmrnn import MLPModel
from ..logic_layer.Commands import *
from ..serializers import UserSerializer

logger = logging.getLogger(__name__)

class UserApiView(viewsets.ViewSet):

    # permission_classes = (IsAuthenticated,)

    @action(methods=['post'], detail=False, permission_classes=[IsAuthenticated, IsAdminUser],
            url_path='add', url_name='add')
    def add(self, request):
        """ Api endpoint to register user"""
        try:
            body = json.loads(request.body.decode('UTF-8'), encoding='UTF-8')
            command = AddUserCommand(body)
            command.execute()

            return JsonResponse({'success':1, 'message':"Usuario Agregado con éxito"})
        except Exception as ex:

            logger.exception("Error")

            return JsonResponse({'success': 0, 'error': _('Ha acurrido un error interno')})


    @action(methods=['get'], detail=False, permission_classes=[IsAuthenticated, IsAdminUser],
            url_path='get', url_name='get')
    def get(self, request):
        """ Api endpoint to register user"""
        try:
            command = GetUsersCommand()
            response = command.execute()

            return Response({'success':1, 'data':response})
        except Exception as ex:

            logger.exception("Error")

            return Response({'success': 0, 'error': _('Ha acurrido un error interno')})

    
    @action(methods=['get'], detail=True, permission_classes=[IsAuthenticated, IsAdminUser],
            url_path='find', url_name='find')
    def get(self, request, pk=None):
        """ Api endpoint to register user"""
        try:

            command = FindUserCommand(pk)
            response = command.execute()

            return Response({'success':1, 'data':response})
        except Exception as ex:

            logger.exception("Error")

            return Response({'success': 0, 'error': _('Ha acurrido un error interno')})