from django.conf.urls import url
from django.urls import path, include
from RD import views

urlpatterns = [
    url(r'^dt_01/', views.rumor_detect_01),
    url(r'^dt_02/', views.rumor_detect_02),
    url(r'^dt_03/', views.rumor_detect_03),
    url(r'^dt_04/', views.rumor_detect_04),
]
