
//图片加载失败后默认图片
function errorImg(_this){
    _this.src = '/static/res/images/imgError.png';
    _this.onerror=null;
}

//身份证正面图片加载失败后默认图片
function errorCardZMImg(_this){
    _this.src = '/static/res/images/cardZMImgError.png';
    _this.onerror=null;
}

//身份证反面图片加载失败后默认图片
function errorCardFMImg(_this){
    _this.src = '/static/res/images/cardFMImgError.png';
    _this.onerror=null;
}

//营业执照图片加载失败后默认图片
function errorYyzzImg(_this){
    _this.src = '/static/res/images/yyzzImgError.png';
    _this.onerror=null;
}

//2018年07月13日 新功能tips提示功能详细用途的提示栏
function getTips(_this, str){
    layer.tips(str , $(_this), {
        tips: [1, '#3595CC'],
        time: 0
    });
}

//2018年07月20日 营业执照、身份证正反面放大看图功能
function seeBigImg(_this){
    var url = $(_this).attr("src");
    layui.use('layer', function(){
        var layer = layui.layer;

        layer.open({
            type: 1,
            area: ['80%', '80%'],
            fixed: false, //不固定
            maxmin: true,
            content: '<img src="'+url+'" width="100%"/>'
        });
    });
}

/***获取当前时间 - 开始***/
function getDate(){
    var today=new Date();
    var y=today.getFullYear();
    var mo=today.getMonth()+1;
    var d=today.getDate();
    var h=today.getHours();
    var m=today.getMinutes();
    var s=today.getSeconds();// 在小于10的数字前加一个‘0’
    mo=checkTime(mo);
    d=checkTime(d);
    h=checkTime(h);
    m=checkTime(m);
    s=checkTime(s);
    return y+"/"+mo+"/"+d+"&nbsp;&nbsp;"+h+":"+m+":"+s;
}
function checkTime(i){
    if (i<10){
        i="0" + i;
    }
    return i;
}
/***获取当前时间 - 结束***/