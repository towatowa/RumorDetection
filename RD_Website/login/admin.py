from django.contrib import admin

# Register your models here.
from . import models

admin.site.site_title = "HEU谣言检测"
admin.site.site_header = "HEU谣言检测后台管理"
admin.site.register(models.User)
