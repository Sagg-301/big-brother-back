from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework import viewsets
from rest_framework.response import Response
from django.utils.translation import gettext as _
import json
import logging
from ..logic_layer.Commands import *
from .validators import *
from ..common.Exceptions.ValidationException import ValidationException

from django.core.paginator import Paginator

logger = logging.getLogger(__name__)
class CrimesApiView(viewsets.ViewSet):

    @action(methods=['get'], detail=False, permission_classes=[IsAuthenticated],
            url_path='stats', url_name='stats')
    def get_stats(self, request):
        """ Api endpoint to register user"""
        try:
            command = CrimesStatCommand()
            response = command.execute()

            return Response({'success':1, 'data':response})
        except Exception as ex:

            logger.exception("Error")

            return Response({'success': 0, 'error': _('Ha acurrido un error interno')})
    @action(methods=['get'], detail=False, permission_classes=[IsAuthenticated],
            url_path='get', url_name='get')
    def get(self, request):
        """ Api endpoint to register user"""
        try:
            command = CrimesGetCommand()
            response = command.execute()
            pages = Paginator(response, request.GET['per_page'] if request.GET['per_page'] else 100)
            page = pages.page(request.GET['page_number'])

            return Response({'success':1, 'data':{
                'total_objects':pages.count,
                'pages': pages.num_pages,
                'crimes':page.object_list,
                'has_next': page.has_next(),
                'has_previous': page.has_previous(),
                'next_page': page.next_page_number()
            }})
        except Exception as ex:

            logger.exception("Error")

            return Response({'success': 0, 'error': _('Ha acurrido un error interno')})

    @action(methods=['post'], detail=False, permission_classes=[IsAuthenticated],
            url_path='add', url_name='add')
    def add(self, request):
        """ Api endpoint to register user"""
        try:
            body = json.loads(request.body.decode('UTF-8'), encoding='UTF-8')

            command = AddCrimeCommand(body)
            response = command.execute()

            return Response({'success':1, 'message':"Crímen agregado con éxito"})
        except Exception as ex:

            logger.exception("Error")

            return Response({'success': 0, 'error': _('Ha acurrido un error interno')})