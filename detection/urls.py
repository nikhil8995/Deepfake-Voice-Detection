from django.urls import path
from .views import PredictView, home

urlpatterns = [
    path('', home, name='home'),
    path('predict/', PredictView.as_view(), name='predict'),
] 