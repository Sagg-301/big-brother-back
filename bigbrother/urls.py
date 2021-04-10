from django.urls import path, include
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from rest_framework_simplejwt import views as jwt_views
from .api.ml import MLApiView
from .api.user import UserApiView
from rest_framework.routers import SimpleRouter

router = SimpleRouter()

router.register(r'api/ml', MLApiView, basename='ml')
router.register(r'api/user', UserApiView, basename='user')

urlpatterns = [
    path('api/token/', jwt_views.TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', jwt_views.TokenRefreshView.as_view(), name='token_refresh'),
]


urlpatterns += router.urls
urlpatterns += staticfiles_urlpatterns()
