from ..Command import Command
from bigbrother.data_layer.Crimes import Crimes as CrimesDAO
from ....serializers import CrimeSerializer
from django.core.paginator import Paginator

class CrimesGetCommand(Command):
    """
    docstring
    """
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        dao = CrimesDAO()

        crimes = dao.get()
        pages = Paginator(crimes, self.payload['per_page'])
        page = pages.page(self.payload['page_number'])

        # return CrimeSerializer(crimes, many=True).data
        return {
                'total_objects':pages.count,
                'pages': pages.num_pages,
                'crimes':CrimeSerializer(page.object_list, many=True).data,
                'has_next': page.has_next(),
                'has_previous': page.has_previous(),
                'next_page': page.next_page_number()
            }