from django.db import models
from login.models import User


# Create your models here.
# 原贴表
class Twitter(models.Model):
    id = models.BigAutoField(primary_key=True)  # 大整数自增
    src_twt = models.TextField(max_length=15 * 1024 * 1024)  # 原贴，最大大小为15M
    label = models.IntegerField(null=True)  # 谣言检测标签
    detect_type = models.CharField(max_length=20, default='unk')
    create_time = models.DateTimeField(auto_now_add=True)  # 创建时间
    last_edit_timestamp = models.DateTimeField(auto_now=True)  # 最后修改时间
    new_add = models.IntegerField(default=1)  # 是否是新添加
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # 用户外键
    user_isExist = models.IntegerField(default=1)  # 用户是否存在
    true_label = models.IntegerField(null=True)  # 谣言真实标签

    class Meta:
        ordering = ["-create_time"]
        verbose_name = "原贴"  # django 后台
        verbose_name_plural = "原贴"  # django 后台


# 评论
class Content(models.Model):
    twitter = models.ForeignKey(Twitter, on_delete=models.CASCADE)  # 级联外键
    text = models.TextField(max_length=10 * 1024 * 1024)  # 谣言文本

    class Meta:
        verbose_name = "评论"  # django 后台
        verbose_name_plural = "评论"  # django 后台


class IMG(models.Model):
    twitter = models.ForeignKey(Twitter, on_delete=models.CASCADE)  # 原贴外键
    image_path = models.ImageField(upload_to='media', default='user1.jpg')  # 图片路径
    name = models.CharField(max_length=64)  # 图片名称

    def __str__(self):
        # 在Python3中使用 def __str__(self):
        return self.name

    class Meta:
        verbose_name = "传播路径图"  # django 后台
        verbose_name_plural = "传播路径图"  # django 后台


class Chinese(models.Model):
    id = models.BigAutoField(primary_key=True)  # 大整数自增
    source_seq = models.TextField(max_length=50 * 1024, default='sentence')  # 原贴，最大大小为1M
    source = models.TextField(max_length=512 * 1024)  # 原贴，最大大小为0.5M
    comment = models.TextField(max_length=6 * 1024 * 1024)  # 评论，最大大小为6M
    label = models.IntegerField(null=True)  # 谣言检测标签
    detect_type = models.CharField(max_length=20, default='unk')
    create_time = models.DateTimeField(auto_now_add=True)  # 创建时间
    last_edit_timestamp = models.DateTimeField(auto_now=True)  # 最后修改时间
    new_add = models.IntegerField(default=1)  # 是否是新添加
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # 用户外键
    user_isExist = models.IntegerField(default=1)  # 用户是否存在
    true_label = models.IntegerField(null=True)  # 谣言真实标签
    image_path = models.ImageField(upload_to='media', default='user1.jpg', null=True)  # 图片路径


    class Meta:
        ordering = ["-create_time"]
        verbose_name = "中文帖子"  # django 后台
        verbose_name_plural = "中文帖子"  # django 后台