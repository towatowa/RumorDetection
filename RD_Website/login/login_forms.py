from django import forms
from captcha.fields import CaptchaField


class UserForm(forms.Form):
    user_id = forms.CharField(label="用户账户", min_length=8, max_length=12, widget=forms.TextInput(
        attrs={'class': 'form-control', 'placeholder': "User ID", 'autofocus': ''}))
    user_password = forms.CharField(label="密码", min_length=6, max_length=12, widget=forms.PasswordInput(
        attrs={'class': 'form-control',  'placeholder': "Password"}))
    captcha = CaptchaField(label='验证码')


class RegisterForm(forms.Form):
    user_id = forms.CharField(label="用户账户", min_length=8, max_length=12, widget=forms.TextInput(
        attrs={'class': 'form-control'}))
    user_name = forms.CharField(label="用户名", max_length=12, widget=forms.TextInput(
        attrs={'class': 'form-control'}))
    user_nickname = forms.CharField(label="用户昵称", max_length=12, widget=forms.TextInput(
        attrs={'class': 'form-control'}))
    password1 = forms.CharField(label="密码", min_length=6, max_length=12, widget=forms.PasswordInput(
        attrs={'class': 'form-control'}))
    password2 = forms.CharField(label="确认密码", min_length=6, max_length=12, widget=forms.PasswordInput(
        attrs={'class': 'form-control'}))
    email = forms.EmailField(label="邮箱地址", widget=forms.EmailInput(attrs={'class': 'form-control'}))
    user_phone = forms.CharField(label="电话号码", min_length=11, max_length=11, widget=forms.TextInput(
        attrs={'class': 'form-control'}))
    captcha = CaptchaField(label='验证码')


class AdminRegisterForm(forms.Form):
    admin_name = forms.CharField(label="用户账号", max_length=15, widget=forms.TextInput(
        attrs={'class': 'form-control'}))
    first_name = forms.CharField(label="First_name", max_length=15, widget=forms.TextInput(
        attrs={'class': 'form-control'}))
    last_name = forms.CharField(label="Last_name", max_length=15,  widget=forms.TextInput(
        attrs={'class': 'form-control'}))
    password1 = forms.CharField(label="密码", min_length=8, max_length=12, widget=forms.PasswordInput(
        attrs={'class': 'form-control'}))
    password2 = forms.CharField(label="确认密码", min_length=8, max_length=12, widget=forms.PasswordInput(
        attrs={'class': 'form-control'}))
    email = forms.EmailField(label="邮箱地址", widget=forms.EmailInput(attrs={'class': 'form-control'}))
    captcha = CaptchaField(label='验证码')