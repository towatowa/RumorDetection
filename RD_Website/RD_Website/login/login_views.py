from django.shortcuts import render, get_object_or_404
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from . import models
from . import login_forms
# Create your views here.
from django.contrib import messages
from django.shortcuts import HttpResponse
from django.contrib import auth  # 引入auth模块
from django.contrib.auth.models import User  # auth应用中引入User类


def index(request):
    if not request.session.get('is_login', None):
        return redirect('/unindex/')  # 未登录进入登录界面
    pass
    return render(request, 'login/index.html')


def un_index(request):
    pass
    return render(request, 'login/un_index.html')


def user_login(request):
    if request.session.get('is_login', None):  # 不允许重复登录
        return redirect('/index/')
    if request.method == 'POST':
        login_form = login_forms.UserForm(request.POST)
        message = '请检查填写的内容！'
        if login_form.is_valid():
            u_id = login_form.cleaned_data.get('user_id')
            u_password = login_form.cleaned_data.get('user_password')

            try:
                user = models.User.objects.get(user_id=u_id)
            except:
                message = '用户不存在！'
                return render(request, 'login/user_login.html', locals())

            if user.user_password == u_password:
                request.session['is_login'] = True  # 向session字典内写入用户状态和数据
                request.session['user_id'] = user.user_id
                request.session['user_name'] = user.user_name
                request.session['user_nickname'] = user.user_nickname
                return redirect('/index/')
            else:
                message = '密码不正确！'
                return render(request, 'login/user_login.html', locals())
        else:
            return render(request, 'login/user_login.html', locals())

    login_form = login_forms.UserForm()
    return render(request, 'login/user_login.html', locals())


def user_register(request):
    if request.session.get('is_login', None):
        return redirect('/index/')
    if request.method == 'POST':
        register_form = login_forms.RegisterForm(request.POST)
        message = "请检查填写的内容！"
        if register_form.is_valid():
            u_id = register_form.cleaned_data.get('user_id')
            u_name = register_form.cleaned_data.get('user_name')
            u_nickname = register_form.cleaned_data.get('user_nickname')
            password1 = register_form.cleaned_data.get('password1')
            password2 = register_form.cleaned_data.get('password2')
            u_phone = register_form.cleaned_data.get('user_phone')
            email = register_form.cleaned_data.get('email')

            if password1 != password2:
                message = '两次输入的密码不同！'
                return render(request, 'login/user_register.html', locals())
            else:
                same_name_user = models.User.objects.filter(user_id=u_id)
                if same_name_user:
                    message = '该账号已存在!'
                    return render(request, 'login/user_register.html', locals())
                same_phone_user = models.User.objects.filter(user_phone=u_phone)
                if same_phone_user:
                    message = '该电话号已经被注册了！'
                    return render(request, 'login/user_register.html', locals())
                same_email_user = models.User.objects.filter(user_email=email)
                if same_email_user:
                    message = '该邮箱已经被注册了！'
                    return render(request, 'login/user_register.html', locals())


                new_user = models.User()
                new_user.user_id = u_id
                new_user.user_name = u_name
                new_user.user_nickname = u_nickname
                new_user.user_phone = u_phone
                new_user.user_password = password1
                new_user.user_email = email
                new_user.save()

                return redirect('/user_login/')
        else:
            return render(request, 'login/user_register.html', locals())
    register_form = login_forms.RegisterForm()
    return render(request, 'login/user_register.html', locals())


def logout(request):
    if not request.session.get('is_login', None):
        # 如果本来就未登录，也就没有登出一说
        return redirect("/unindex/")
    request.session.flush()
    # 或者使用下面的方法
    # del request.session['is_login']
    # del request.session['user_id']
    # del request.session['user_name']
    return redirect("/unindex/")


def login_new(request):
    if request.session.get('is_login', None):  # 不允许重复登录
        return redirect('/index_new/')
    if request.method == 'POST':
        u_id = request.POST.get('user_id')
        u_password = request.POST.get('user_password')
        message = '请检查填写的内容！'
        try:
            user = models.User.objects.get(user_id=u_id)
        except:
            message = '该用户不存在！'
            return render(request, 'login/login1.html', locals())
        if user.user_password == u_password:
            request.session['is_login'] = True  # 向session字典内写入用户状态和数据
            request.session['user_id'] = user.user_id
            request.session['user_name'] = user.user_name
            request.session['user_nickname'] = user.user_nickname
            return redirect('/index_new/')
        else:
            message = '密码不正确！'
            return render(request, 'login/login1.html', locals())
    else:
        return render(request, 'login/login1.html', locals())

    # login_form = login_forms.UserForm()
    # return render(request, 'login/login1.html')


def frame(request):
    if not request.session.get('is_login', None):
        return redirect('/unindex_new/')  # 未登录进入登录界面
    return render(request, 'login/frame.html')


def rumor_detect(request):
    pass
    return render(request, 'login/second/rumor_detection.html')


def index_inside(request):
    pass
    return render(request, 'login/second/index_inside.html')


def unindex_new(request):
    pass
    return render(request, 'login/unindex_new.html')


def register_new(request):
    if request.session.get('is_login', None):
        return redirect('/index_new/')
    if request.method == 'POST':
        register_form = login_forms.RegisterForm(request.POST)
        message = "请检查填写的内容！"
        if register_form.is_valid():
            u_id = register_form.cleaned_data.get('user_id')
            u_name = register_form.cleaned_data.get('user_name')
            u_nickname = register_form.cleaned_data.get('user_nickname')
            password1 = register_form.cleaned_data.get('password1')
            password2 = register_form.cleaned_data.get('password2')
            u_phone = register_form.cleaned_data.get('user_phone')
            email = register_form.cleaned_data.get('email')

            if password1 != password2:
                message = '两次输入的密码不同！'
                return render(request, 'login/register_new.html', locals())
            else:
                same_name_user = models.User.objects.filter(user_id=u_id)
                if same_name_user:
                    message = '该账号已存在!'
                    return render(request, 'login/register_new.html', locals())
                same_phone_user = models.User.objects.filter(user_phone=u_phone)
                if same_phone_user:
                    message = '该电话号已经被注册了！'
                    return render(request, 'login/register_new.html', locals())
                same_email_user = models.User.objects.filter(user_email=email)
                if same_email_user:
                    message = '该邮箱已经被注册了！'
                    return render(request, 'login/register_new.html', locals())

                new_user = models.User()
                new_user.user_id = u_id
                new_user.user_name = u_name
                new_user.user_nickname = u_nickname
                new_user.user_phone = u_phone
                new_user.user_password = password1
                new_user.user_email = email
                new_user.save()

                return redirect('/login_new/')
        else:
            return render(request, 'login/register_new.html', locals())
    register_form = login_forms.RegisterForm()
    return render(request, 'login/register_new.html', locals())


def admin_register(request):
    if request.session.get('is_login', None):
        return redirect('/unindex_new/')
    if request.method == 'POST':
        admin_form = login_forms.AdminRegisterForm(request.POST)
        message = "请检查填写的内容！"
        if admin_form.is_valid():
            admin_name = admin_form.cleaned_data.get('admin_name')
            first_name = admin_form.cleaned_data.get('first_name')
            last_name = admin_form.cleaned_data.get('last_name')
            password1 = admin_form.cleaned_data.get('password1')
            password2 = admin_form.cleaned_data.get('password2')
            email = admin_form.cleaned_data.get('email')
            if admin_name == password1:
                message = '密码不能和用户名相同'
                return render(request, 'login/admin_register.html', locals())
            num = 0
            for cha in password1:
                if cha in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    num += 1
            if len(password1) == num:
                message = '密码不能全部为数字'
                return render(request, 'login/admin_register.html', locals())
            if password1 != password2:
                message = '两次输入的密码不同！'
                return render(request, 'login/admin_register.html', locals())
            else:
                same_name_user = auth.models.User.objects.filter(username=admin_name)
                if same_name_user:
                    message = '该账号已存在!'
                    return render(request, 'login/admin_register.html', locals())
                same_email_user = auth.models.User.objects.filter(email=email)
                if same_email_user:
                    message = '该邮箱已经被注册了！'
                    return render(request, 'login/admin_register.html', locals())

                new_user = auth.models.User()
                new_user.username = admin_name
                new_user.first_name = first_name
                new_user.last_name = last_name
                new_user.is_staff = 0
                new_user.is_active = 0
                new_user.password = password1
                new_user.email = email
                new_user.save()

                return redirect('/unindex_new/')
        else:
            return render(request, 'login/admin_register.html', locals())
    admin_form = login_forms.AdminRegisterForm()
    return render(request, 'login/admin_register.html', locals())


def logout_new(request):
    if not request.session.get('is_login', None):
        # 如果本来就未登录，也就没有登出一说
        return redirect("/unindex_new/")
    request.session.flush()
    # 或者使用下面的方法
    # del request.session['is_login']
    # del request.session['user_id']
    # del request.session['user_name']
    return redirect("/unindex_new/")


def personal_center(request):
    user_table = login_forms.RegisterForm()
    user_id = request.session['user_id']
    user = models.User.objects.get(user_id=user_id)
    print(user)
    return render(request, 'login/second/personal _center.html',locals())


def personal_setting(request):
    pass
    return render(request, 'login/second/personal _setting.html')
