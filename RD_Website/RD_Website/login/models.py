from django.db import models


# Create your models here.
class User(models.Model):
    user_id = models.CharField(max_length=128, primary_key=True, verbose_name='用户账号')
    user_name = models.CharField(max_length=20, verbose_name='用户名')
    user_nickname = models.CharField(max_length=20, verbose_name='昵称')
    user_password = models.CharField(max_length=128, verbose_name='密码')
    user_email = models.EmailField(unique=True, verbose_name='邮箱')
    user_phone = models.CharField(max_length=11, verbose_name='电话号码')
    user_table = models.CharField(max_length=11, verbose_name='谣言表')
    user_c_time = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')

    def __str__(self):
        return self.user_id

    class Meta:
        ordering = ["-user_c_time"]
        verbose_name = "普通用户"  # django 后台
        verbose_name_plural = "普通用户"  # django 后台

