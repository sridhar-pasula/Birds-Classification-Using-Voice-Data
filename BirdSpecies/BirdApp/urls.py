from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
	       path('UserLogin', views.UserLogin, name="UserLogin"),
	       path('UserLoginAction', views.UserLoginAction, name="UserLoginAction"),	   
	       path('UploadVoice', views.UploadVoice, name="UploadVoice"),
	       path('UploadVoiceAction', views.UploadVoiceAction, name="UploadVoiceAction"),
	       
]