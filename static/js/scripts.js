$(function () {

    // init feather icons
    feather.replace();

    // init tooltip & popovers
    $('[data-toggle="tooltip"]').tooltip();
    $('[data-toggle="popover"]').popover();

    // page scroll
    $('a.page-scroll').bind('click', function (event) {
        var $anchor = $(this);
        $('html, body').stop().animate({
            scrollTop: $($anchor.attr('href')).offset().top - 20
        }, 1000);
        event.preventDefault();
    });

    // slick slider
    $('.slick-about').slick({
        slidesToShow: 1,
        slidesToScroll: 1,
        autoplay: true,
        autoplaySpeed: 3000,
        dots: true,
        arrows: false
    });

    //toggle scroll menu
    var scrollTop = 0;
    $(window).scroll(function () {
        var scroll = $(window).scrollTop();
        //adjust menu background
        if (scroll > 80) {
            if (scroll > scrollTop) {
                $('.smart-scroll').addClass('scrolling').removeClass('up');
            } else {
                $('.smart-scroll').addClass('up');
            }
        } else {
            // remove if scroll = scrollTop
            $('.smart-scroll').removeClass('scrolling').removeClass('up');
        }

        scrollTop = scroll;

        // adjust scroll to top
        if (scroll >= 600) {
            $('.scroll-top').addClass('active');
        } else {
            $('.scroll-top').removeClass('active');
        }
        return false;
    });

    // scroll top top
    $('.scroll-top').click(function () {
        $('html, body').stop().animate({
            scrollTop: 0
        }, 1000);
    });

    $('#show-image-button').click(function () {
        // Toggle the visibility of the camera stream div
        var cameraStreamDiv = $('#camera-stream-div');
        cameraStreamDiv.toggle();
    });

// Picture button click event
$("#picture-upload").change(function (e) {
    // 获取用户选择的文件
    var file = e.target.files[0];

    // 检查文件类型
    if (file.type.startsWith('image/')) {
        // 如果是图片文件，创建一个 FileReader 对象来读取文件内容
        var reader = new FileReader();

        // 定义 FileReader 的加载完成事件处理程序
        reader.onload = function (event) {
            // 创建一个新的图像元素
            var image = document.createElement('img');
            image.src = event.target.result;

            // 添加 CSS 类以限制图像大小
            image.classList.add('preview-image');

            // 找到 new-content-area 元素
            var newContentArea = document.getElementById('new-content-area');

            // 清空内容
            newContentArea.innerHTML = '';

            // 添加图像到 new-content-area
            newContentArea.appendChild(image);
        };

        // 读取文件内容
        reader.readAsDataURL(file);
    } else {
        alert('请选择图片文件');
    }
});






$(document).ready(function () {
    // 设置初始按钮状态
    $('#picture-button').addClass('active');

    // 处理按钮点击事件
    $('#picture-button, #video-button, #camera-button').click(function () {
        // 移除所有按钮的 active 类
        $('#picture-button, #video-button, #camera-button').removeClass('active');

        // 清空 new-content-area 的内容
        var newContentArea = document.getElementById('new-content-area');
        newContentArea.innerHTML = '';

        // 清空已上传的文件
        $("#picture-upload").val(''); // 清空图片文件输入
        $("#video-upload").val(''); // 清空视频文件输入

        // 将点击的按钮设为 active
        $(this).addClass('active');

        // 处理每个按钮的点击事件
        if ($(this).is('#picture-button')) {
            $("#picture-upload").show();
            $("#video-upload").hide();
            $("#camera-stream-div2").hide();
            $("#result-preview-text").text("Picture result preview");
            $("#convert-button").show();
            // $("#picture-preview").show();
            $("#video-preview").hide();
        } else if ($(this).is('#video-button')) {
            $("#video-upload").show();
            $("#picture-upload").hide();
            $("#camera-stream-div2").hide();
            $("#result-preview-text").text("Video result preview");
            $("#convert-button").show();
            $("#picture-preview").hide();
            // $("#video-preview").show();
        } else if ($(this).is('#camera-button')) {
            $("#picture-upload").hide();
            $("#video-upload").hide();
            $("#result-preview-text").text("Real-time result preview");
            $("#convert-button").hide();
            $("#camera-stream-div2").show();
            $("#picture-preview").hide();
            $("#video-preview").hide();
        }
        
    });
    $('#picture-button').click();
    $('#picture-preview').hide();

});

// 修改 Convert 按钮的点击事件
$('#convert-button').click(function() {
    // 创建一个 FormData 对象
    var formData = new FormData();

    if ($('#picture-upload')[0].files.length > 0) {
        // 用户上传了图片文件
        formData.append('picture', $('#picture-upload')[0].files[0]);
        $("#picture-preview").show();
    } else if ($('#video-upload')[0].files.length > 0) {
        // 用户上传了视频文件
        formData.append('video', $('#video-upload')[0].files[0]);
        $("#video-preview").show();
    } else {
        alert('请选择图片或视频文件');
        return;
    }

    $.ajax({
        type: 'POST',
        url: '/process',
        data: formData,
        contentType: false,
        processData: false,
        success: function(data) {
            // 显示处理后的图片和视频
            $('#picture-preview').attr('src', data.picture_preview);
            $('#video-preview').attr('src', data.video_preview);
        }
    });
});

    
});
