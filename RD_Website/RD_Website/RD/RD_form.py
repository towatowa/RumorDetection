from django import forms


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