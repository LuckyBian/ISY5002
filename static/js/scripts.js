$(function () {
    // 初始化 feather icons
    feather.replace();

    // 初始化 tooltip & popovers
    $('[data-toggle="tooltip"]').tooltip();
    $('[data-toggle="popover"]').popover();

    // 页面滚动
    $('a.page-scroll').bind('click', function (event) {
        var $anchor = $(this);
        $('html, body').stop().animate({
            scrollTop: $($anchor.attr('href')).offset().top - 20
        }, 1000);
        event.preventDefault();
    });

    // Slick 轮播
    $('.slick-about').slick({
        slidesToShow: 1,
        slidesToScroll: 1,
        autoplay: true,
        autoplaySpeed: 3000,
        dots: true,
        arrows: false
    });

    // 切换滚动菜单
    var scrollTop = 0;
    $(window).scroll(function () {
        var scroll = $(window).scrollTop();
        // 调整菜单背景
        if (scroll > 80) {
            if (scroll > scrollTop) {
                $('.smart-scroll').addClass('scrolling').removeClass('up');
            } else {
                $('.smart-scroll').addClass('up');
            }
        } else {
            // 如果滚动位置等于顶部则移除类
            $('.smart-scroll').removeClass('scrolling').removeClass('up');
        }

        scrollTop = scroll;

        // 调整滚动到顶部按钮
        if (scroll >= 600) {
            $('.scroll-top').addClass('active');
        } else {
            $('.scroll-top').removeClass('active');
        }
        return false;
    });

    // 滚动到顶部
    $('.scroll-top').click(function () {
        $('html, body').stop().animate({
            scrollTop: 0
        }, 1000);
    });

    $('#show-image-button').click(function () {
        // 切换相机流 div 的可见性
        var cameraStreamDiv = $('#camera-stream-div');
        cameraStreamDiv.toggle();
    });

    // 图片按钮点击事件
    $("#picture-upload").change(function (e) {
        var file = e.target.files[0];
        if (file.type.startsWith('image/')) {
            var reader = new FileReader();
            reader.onload = function (event) {
                var image = document.createElement('img');
                image.src = event.target.result;
                image.classList.add('preview-image');
                var newContentArea = document.getElementById('new-content-area');
                newContentArea.innerHTML = '';
                newContentArea.appendChild(image);
            };
            reader.readAsDataURL(file);
        } else {
            alert('Please upload an image file');
        }
    });

    $("#video-upload").change(function (e) {
        var file = e.target.files[0];
        if (file.type.startsWith('video/')) {
            var reader = new FileReader();
            reader.onload = function (event) {
                var video = document.createElement('video');
                video.src = event.target.result;
                video.controls = true;
                video.classList.add('preview-image');
                var newContentArea = document.getElementById('new-content-area');
                newContentArea.innerHTML = '';
                newContentArea.appendChild(video);
            };
            reader.readAsDataURL(file);
        } else {
            alert('Please upload a video file');
        }
    });
    

    $(document).ready(function () {
        $('#picture-button').addClass('active');
        $('#picture-button, #video-button, #camera-button').click(function () {
            $('#picture-button, #video-button, #camera-button').removeClass('active');
            var newContentArea = document.getElementById('new-content-area');
            newContentArea.innerHTML = '';
            $("#picture-upload").val('');
            $("#video-upload").val('');
            $(this).addClass('active');
            if ($(this).is('#picture-button')) {
                $("#picture-upload").show();
                $("#video-upload").hide();
                $("#camera-stream-div2").hide();
                $("#result-preview-text").text("Picture result preview");
                $("#convert-button").show();
                $("#picture-preview").hide();
                $("#video-preview").hide();
            } else if ($(this).is('#video-button')) {
                $("#video-upload").show();
                $("#picture-upload").hide();
                $("#camera-stream-div2").hide();
                $("#result-preview-text").text("Video result preview");
                $("#convert-button").show();
                $("#picture-preview").hide();
                $("#video-preview").hide();
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


    $('#convert-button').click(function() {
        // 创建一个 FormData 对象
        var formData = new FormData();
    
        if ($('#picture-upload')[0].files.length > 0) {
            // 用户上传了图片文件
            formData.append('picture', $('#picture-upload')[0].files[0]);
            $("#picture-preview").show();
    
            $.ajax({
                type: 'POST',
                url: '/process',  // 使用处理图片的端口
                data: formData,
                contentType: false,
                processData: false,
                success: function(data) {
                    // 从响应数据中获取宽度和高度信息
                    var imageWidth = data.image_width;
                    var imageHeight = data.image_height;
            
                    // 获取图片元素
                    var picturePreview = document.getElementById("picture-preview");
            
                    // 将宽度和高度信息应用于生成图像
                    picturePreview.src = data.picture_preview;
                    picturePreview.width = imageWidth;
                    picturePreview.height = imageHeight;
            
                    // 显示处理后的图片
                    $('#picture-preview').attr('src', data.picture_preview);
                }
            });
        } else if ($('#video-upload')[0].files.length > 0) {
            // 用户上传了视频文件
            formData.append('video', $('#video-upload')[0].files[0]);
            
    
            $.ajax({
                type: 'POST',
                url: '/process_video',  // 使用处理视频的端口
                data: formData,
                contentType: false,
                processData: false,
                success: function(data) {
                    // 处理视频的响应
                    // 根据需要在这里添加视频的处理逻辑
                    $("#video-preview").show();
                    $('#video-preview').attr('src', data.processed_video_path);
                    
                }
            });
        } else {
            alert('请选择图片或视频文件');
            return;
        }
    });
});
