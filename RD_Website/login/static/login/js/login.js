$(function(){
	
	init();
	$(window).resize(function(){
		LIULANQI_RESIZE = 1;
		init();
	});
	
})

//初始化页面
var LIULANQI_RESIZE = 0;
var myCarousel = null;
function init(){
	var win_h = $(window).height();
	
	
	if(win_h >= 900){
		$(".form_tzgg").height(win_h - 148 - 36);
	}else if(win_h >= 768 && win_h <= 899){
		$(".form_tzgg").height(win_h - 129 - 36);
	}else if(win_h <= 767){
		$(".form_tzgg").height(win_h - 93 - 36);
	}

	var options = {
        elem: '#loginLbt',
        width: '100%', //设置容器宽度
        height: win_h,
        interval: 4000,
//			trigger:'hover',
        indicator: 'inside',
        arrow: 'none' //始终显示箭头
    };

	if(LIULANQI_RESIZE == 0){
        layui.use('carousel', function() {
            var carousel = layui.carousel;
            //建造实例
            myCarousel = carousel.render(options);
        });
	}else{
        myCarousel.reload(options);
	}
	
}

//获取光标
function getFocus(_this){
	$(_this).find("input").focus()
}
