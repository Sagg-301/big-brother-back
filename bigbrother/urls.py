from django.urls import path
from django.contrib import admin
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from . import api

urlpatterns = [
    path('train_mlm', api.ml.train_mlp),
    path('predict/', api.ml.predict),
]

urlpatterns += staticfiles_urlpatterns()
