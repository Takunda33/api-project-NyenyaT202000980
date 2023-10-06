from django.urls import path, include
from . import views

urlpatterns = [
   path("",views.predict_sneakers),
   path("predict/", views.predict_sneakers),
   path('form/', views.api_view, name='form'),

]