$(function () {

    feather.replace();

    $('[data-toggle="tooltip"]').tooltip();
    $('[data-toggle="popover"]').popover();


    $('a.page-scroll').bind('click', function (event) {
        var $anchor = $(this);
        $('html, body').stop().animate({
            scrollTop: $($anchor.attr('href')).offset().top - 20
        }, 1000);
        event.preventDefault();
    });


    $('.slick-about').slick({
        slidesToShow: 1,
        slidesToScroll: 1,
        autoplay: true,
        autoplaySpeed: 3000,
        dots: true,
        arrows: false
    });


    var scrollTop = 0;
    $(window).scroll(function () {
        var scroll = $(window).scrollTop();

        if (scroll > 80) {
            if (scroll > scrollTop) {
                $('.smart-scroll').addClass('scrolling').removeClass('up');
            } else {
                $('.smart-scroll').addClass('up');
            }
        } else {
            $('.smart-scroll').removeClass('scrolling').removeClass('up');
        }

        scrollTop = scroll;


        if (scroll >= 600) {
            $('.scroll-top').addClass('active');
        } else {
            $('.scroll-top').removeClass('active');
        }
        return false;
    });

    $('.scroll-top').click(function () {
        $('html, body').stop().animate({
            scrollTop: 0
        }, 1000);
    });

    $('#show-image-button').click(function () {
        var cameraStreamDiv = $('#camera-stream-div');
        cameraStreamDiv.toggle();
    });


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
                $("#enable-camera-button").hide();
                $("#detect-button").hide();
                $("#camera-stream").hide();
            } else if ($(this).is('#video-button')) {
                $("#video-upload").show();
                $("#picture-upload").hide();
                $("#camera-stream-div2").hide();
                $("#result-preview-text").text("Video result preview");
                $("#convert-button").show();
                $("#picture-preview").hide();
                $("#video-preview").hide();
                $("#enable-camera-button").hide();
                $("#detect-button").hide();
                $("#camera-stream").hide();
            } else if ($(this).is('#camera-button')) {
                $("#enable-camera-button").show();
                $("#detect-button").show();
                $("#picture-upload").hide();
                $("#video-upload").hide();
                $("#result-preview-text").text("Real-time result preview");
                $("#convert-button").hide();
                $("#camera-stream-div2").show();
                $("#picture-preview").hide();
                $("#video-preview").hide();
                $("#camera-stream").hide();
            }
        });
        $('#picture-button').click();
        $('#picture-preview').hide();
    });

    $('#enable-camera-button').click(function() {
        $("#camera-stream").show();
        $("#detect-button").show();

    })

    $('#stop-button').click(function() {
        $("#detect-button").show();
        $("#stop-button").hide();

    })

    $('#detect-button').click(function() {
        $("#stop-button").show();
        $("#detect-button").hide();

    })


    $('#convert-button').click(function() {

        var formData = new FormData();
    
        if ($('#picture-upload')[0].files.length > 0) {

            formData.append('picture', $('#picture-upload')[0].files[0]);
            $("#picture-preview").show();
    
            $.ajax({
                type: 'POST',
                url: '/process',  
                data: formData,
                contentType: false,
                processData: false,
                success: function(data) {

                    var imageWidth = data.image_width;
                    var imageHeight = data.image_height;
            

                    var picturePreview = document.getElementById("picture-preview");
            

                    picturePreview.src = data.picture_preview;
                    picturePreview.width = imageWidth;
                    picturePreview.height = imageHeight;
            

                    $('#picture-preview').attr('src', data.picture_preview);
                }
            });
        } else if ($('#video-upload')[0].files.length > 0) {

            formData.append('video', $('#video-upload')[0].files[0]);
            
    
            $.ajax({
                type: 'POST',
                url: '/process_video',  
                data: formData,
                contentType: false,
                processData: false,
                success: function(data) {
                    $("#video-preview").show();
                    $('#video-preview').attr('src', data.processed_video_path);
                    
                }
            });
        } else {
            alert('Please upload a picture/video file.');
            return;
        }
    });
});
