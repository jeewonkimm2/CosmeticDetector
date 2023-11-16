from django.urls import path
from . import views

app_name = 'image_processing'

urlpatterns = [
    path('process_video', views.process_video, name= 'process_video'),
    
]