$(function(){
	
})

//上一步下一步
function timePageClick(_this,tag){
	var con = $(_this).parent().parent().parent();
	var step = con.index();
	if(tag == "+"){
		$(".timePage .time ul").children().eq(step).find(".iconfont").attr("class","iconfont icon-duigou").text("");
		$(".timePage .time ul").children().eq(step).find("font").css({
			"font-weight": "normal",
			"color": "#666"
		});
		step++;
		//改变时间导航的样式
		$(".timePage .time ul").children().eq(step).addClass("active");
	}else{
		$(".timePage .time ul").children().eq(step).removeClass("active");
		step--;
		$(".timePage .time ul").children().eq(step).addClass("active").find(".iconfont").attr("class","iconfont").text(step+1);
		$(".timePage .time ul").children().eq(step).find("font").css({
			"font-weight": "600",
			"color": "#000"
		});
	}
	$(".timePage .page form").children().eq(step).siblings().hide();
	$(".timePage .page form").children().eq(step).show();
}
