"""RD_Website URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from login import login_views
path_base_01 = 'index_new/'
urlpatterns = [
    path('user_login/', login_views.user_login),  # 登录
    path('user_register/', login_views.user_register),  # 注册
    path('index/', login_views.index),  # 主页
    path('unindex/', login_views.un_index),  # 未登录主页
    path('logout/', login_views.logout),  # 登出
    path('captcha/', include('captcha.urls')),


    path('login_new/', login_views.login_new),  # 登录
    path('index_new/', login_views.frame),  # 主页
    path('rumor_detect/', login_views.rumor_detect),  # 检测页面
    path('index_inside/', login_views.index_inside),  # 内主页
    path('', login_views.unindex_new),  # 未登录主页
    path('unindex_new/', login_views.unindex_new),  # 未登录主页
    path('logout_new/', login_views.logout_new),  # 登出
    path('register_new/', login_views.register_new),  # 用户注册
    path('admin_register/', login_views.admin_register),  # 管理员注册
    path('personal_center/', login_views.personal_center),  # 个人中心
    path('personal_setting/', login_views.personal_setting),  # 个人设置


]
