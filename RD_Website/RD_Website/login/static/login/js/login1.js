$(function(){
	
	init();
	$(window).resize(function(){
		init();
	});
	
})

//初始化页面
function init(){
	var win_h = $(window).height();
	var win_w = $(window).width();
	
	$(".loginBg").css({
		"height": win_h/2+"px",
		"background-size": win_w+"px "+win_h/2+"px"
	});
	$(".login_main").height(win_h);
	$(".login").css("margin-top","-"+$(".login").height()/2+"px");
}
