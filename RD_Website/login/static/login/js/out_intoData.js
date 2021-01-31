//根据2018.03.01日 张家港客户需求改版的要求，在新的后台框架中提供以下功能

//新增
function addOutDataList(id){
	var str = '<tr>'+
		'<td>'+
		    '<select name="modules" lay-verify="required" lay-search="">'+
	          '<option value="">直接选择或搜索选择</option>'+
	          '<option value="1">layer</option>'+
	          '<option value="2">form</option>'+
	          '<option value="3">layim</option>'+
	          '<option value="4">element</option>'+
	          '<option value="5">laytpl</option>'+
	          '<option value="6">upload</option>'+
	          '<option value="7">laydate</option>'+
	          '<option value="8">laypage</option>'+
	          '<option value="9">flow</option>'+
	          '<option value="10">util</option>'+
	          '<option value="11">code</option>'+
	          '<option value="12">tree</option>'+
	          '<option value="13">layedit</option>'+
	          '<option value="14">nav</option>'+
	          '<option value="15">tab</option>'+
	          '<option value="16">table</option>'+
	          '<option value="17">select</option>'+
	          '<option value="18">checkbox</option>'+
	          '<option value="19">switch</option>'+
	          '<option value="20">radio</option>'+
	        '</select>'+
		'</td>'+
		'<td>'+
		    '<input type="tel" name="phone" lay-verify="required" autocomplete="off" class="layui-input">'+
		'</td>'+
		'<td>'+
		    '<input type="tel" name="phone" lay-verify="required" autocomplete="off" class="layui-input">'+
		'</td>'+
		'<td>'+
		    '<input type="tel" name="phone" lay-verify="required" autocomplete="off" class="layui-input">'+
		'</td>'+
		'<td>'+
		    '<input type="tel" name="phone" lay-verify="required" autocomplete="off" class="layui-input">'+
		'</td>'+
		'<td>'+
		    '<input type="tel" name="phone" lay-verify="required" autocomplete="off" class="layui-input">'+
		'</td>'+
		'<td>'+
		    '<input type="tel" name="phone" lay-verify="required" autocomplete="off" class="layui-input">'+
		'</td>'+
		'<td>'+
		    '<input type="tel" name="phone" lay-verify="required" autocomplete="off" class="layui-input">'+
		'</td>'+
		'<td>'+
		    '<input type="tel" name="phone" lay-verify="required" autocomplete="off" class="layui-input">'+
		'</td>'+
		'<td>'+
		    '<input type="tel" name="phone" lay-verify="required" autocomplete="off" class="layui-input">'+
		'</td>'+
		'<td>'+
			'<a class="layui-btn layui-btn-xs" onclick="delCustomList(this)">删除</a>'+
		'</td>'+
	'</tr>';
	$("#"+id).append(str);
	layui.use(['form','laydate'], function() {
		var form = layui.form;
		
		form.render();
	});
	//重新初始化Iframe的高度
	cframeInit();
}

//新增
function addIntoDataList(id){
	var str = '<tr>'+
		'<td>'+
		    '<select name="modules" lay-verify="required" lay-search="">'+
	          '<option value="">直接选择或搜索选择</option>'+
	          '<option value="1">layer</option>'+
	          '<option value="2">form</option>'+
	          '<option value="3">layim</option>'+
	          '<option value="4">element</option>'+
	          '<option value="5">laytpl</option>'+
	          '<option value="6">upload</option>'+
	          '<option value="7">laydate</option>'+
	          '<option value="8">laypage</option>'+
	          '<option value="9">flow</option>'+
	          '<option value="10">util</option>'+
	          '<option value="11">code</option>'+
	          '<option value="12">tree</option>'+
	          '<option value="13">layedit</option>'+
	          '<option value="14">nav</option>'+
	          '<option value="15">tab</option>'+
	          '<option value="16">table</option>'+
	          '<option value="17">select</option>'+
	          '<option value="18">checkbox</option>'+
	          '<option value="19">switch</option>'+
	          '<option value="20">radio</option>'+
	        '</select>'+
		'</td>'+
		'<td>'+
		    '<input type="tel" name="phone" lay-verify="required" autocomplete="off" class="layui-input">'+
		'</td>'+
		'<td>'+
		    '<input type="tel" name="phone" lay-verify="required" autocomplete="off" class="layui-input">'+
		'</td>'+
		'<td>'+
		    '<input type="tel" name="phone" lay-verify="required" autocomplete="off" class="layui-input">'+
		'</td>'+
		'<td>'+
		    '<input type="tel" name="phone" lay-verify="required" autocomplete="off" class="layui-input">'+
		'</td>'+
		'<td>'+
		    '<input type="tel" name="phone" lay-verify="required" autocomplete="off" class="layui-input">'+
		'</td>'+
		'<td>'+
		    '<input type="tel" name="phone" lay-verify="required" autocomplete="off" class="layui-input">'+
		'</td>'+
		'<td>'+
		    '<input type="tel" name="phone" lay-verify="required" autocomplete="off" class="layui-input">'+
		'</td>'+
		'<td>'+
		    '<input type="tel" name="phone" lay-verify="required" autocomplete="off" class="layui-input">'+
		'</td>'+
		'<td>'+
		    '<input type="tel" name="phone" lay-verify="required" autocomplete="off" class="layui-input">'+
		'</td>'+
		'<td>'+
			'<a class="layui-btn layui-btn-xs" onclick="delCustomList(this)">删除</a>'+
		'</td>'+
	'</tr>';
	$("#"+id).append(str);
	layui.use(['form','laydate'], function() {
		var form = layui.form;
		
		form.render();
	});
	//重新初始化Iframe的高度
	cframeInit();
}

//删除
function delCustomList(_this){
	layui.use(['form','laydate'], function() {
		layer.confirm('确定要删除么？', {
			btn: ['确定', '取消'] //按钮
		}, function() {
			$(_this).parent().parent().remove();
			layer.msg('删除成功', {
				icon: 1
			});
		}, function() {
			layer.msg('取消删除', {
				time: 2000 //20s后自动关闭
			});
		});
	});
}
