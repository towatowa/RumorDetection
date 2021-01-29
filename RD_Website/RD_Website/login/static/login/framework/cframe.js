$(function(){
	
	$(window).load(function(){
		cframeInit();
		//后台首页的全屏功能的bug，如果不得不取出Iframe的大小变化后的监听事件。
		$(window).resize(function(){
			cframeInit();
		});
	});
	
	//公共form表单验证
    layui.use('form', function() {
        var form = layui.form;
        form.verify({
            //数组的两个值分别代表：[正则匹配、匹配不符时的提示文字]

            //只能输入中文
            ZHCheck: [
                /^[\u0391-\uFFE5]+$/
                ,'只允许输入中文'
            ],
			//先非空，如果有内容则验证邮箱格式
            NEWEmail: function(value, item){ //value：表单的值、item：表单的DOM对象
                if(value != null && value != ""){
                    if(!/^\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*$/.test(value)){
                        return '邮箱格式不正确';
                    }
                }
            },
			//校验登录名是否重复
            LoginName:function(value,item){
                var tag = submitcheck(value);
                if(!tag){
                    return "用户名已存在";
                }
            },
			//验证价格数字
			//^[1-9]\d*(.\d{1,2})?$ ： 1-9开头，后跟是0-9，可以跟小数点，但小数点后要带上1-2位小数，类似2,2.0,2.1,2.22等
			//^0(.\d{1,2})?$ ： 0开头，后可以跟小数点，小数点后要待上1-2位小数，类似0,0.22,0.1等
            PriceCheck:[
                /(^[1-9]\d*(\.\d{1,2})?$)|(^0(\.\d{1,2})?$)/
                ,'请按照标准价格格式输入，小数位可选择输入两位'
            ]
			//身份证号
//          identity: function(value, item){ //value：表单的值、item：表单的DOM对象
//          	var reg1 = /^[1-9]\d{5}(18|19|([23]\d))\d{2}((0[1-9])|(10|11|12))(([0-2][1-9])|10|20|30|31)\d{3}[0-9Xx]$/;
//          	var reg2 = /^[1-9]\d{5}\d{2}((0[1-9])|(10|11|12))(([0-2][1-9])|10|20|30|31)\d{2}[0-9Xx]$/;
//              if(!reg1.test(value) $$ !reg2.test(value)){
//                  return '请输入正确的身份证号......';
//              }
//          }

        });
    });
})

//初始化页面
function cframeInit(){
	var win_h = $(window).height();
	var conheight;
	try{
		conheight = $('#mainIframe',parent.document).parents(".frameMain").find(".con").height();
	}catch(e){
		conheight = win_h;
	}
	
	var fullScreenTAG = sessionStorage.getItem("fullScreenTAG");
	if(fullScreenTAG == 1){
		conheight = $(parent.document).height();
	}
	
	//计算Iframe的高度与父元素相同
	$('#mainIframe',parent.document).css("height",conheight);
	//计算Iframe内cBody的高度，使其固定
	$(".cBody").height(conheight - 20);
	//为cBody设置滚动条的样式
	$(".cBody").mCustomScrollbar();
}

//商品图片放大
function imgBig(_this){
    _this.className = "imgBig";
    _this.width = "200";
    _this.height = "200";
}

//商品图片放小
function imgSmall(_this){
    _this.className = "imgSmall";
    _this.width = "20";
    _this.height = "20";
}

//图片加载失败后默认图片
function errorImg(_this){
    _this.src = '../../images/imgError.png';
    _this.onerror=null;
}

//身份证正面图片加载失败后默认图片
function errorCardZMImg(_this){
    _this.src = '../../images/cardZMImgError.png';
    _this.onerror=null;
}

//身份证反面图片加载失败后默认图片
function errorCardFMImg(_this){
    _this.src = '../../images/cardFMImgError.png';
    _this.onerror=null;
}

//营业执照图片加载失败后默认图片
function errorYyzzImg(_this){
    _this.src = '../../images/yyzzImgError.png';
    _this.onerror=null;
}


//2018年07月13日 新功能tips提示功能详细用途的提示栏
function getTips(_this, str){
	layer.tips(str , $(_this), {
	  	tips: [1, '#3595CC'],
	  	time: 0
	});
}