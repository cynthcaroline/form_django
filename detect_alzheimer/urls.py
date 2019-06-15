from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict_json/', views.predict_json, name='predict_json'),
]