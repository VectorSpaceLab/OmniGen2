function switchVideo(prefix, videoContainerId, preview_id) {
    // Reference to all video containers
    var video1Container = document.getElementById(prefix + 'video1Container');
    var video2Container = document.getElementById(prefix + 'video2Container');

    // Hide all video containers first
    video1Container.style.display = 'none';
    video2Container.style.display = 'none';

    // Stop and reset videos
    var videos = video1Container.getElementsByTagName('video');
    for (var i = 0; i < videos.length; i++) {
        videos[i].pause();
    }

    videos = video2Container.getElementsByTagName('video');
    for (var i = 0; i < videos.length; i++) {
        videos[i].pause();
    }

    // Show the selected video container
    var selectedVideoContainer = document.getElementById(prefix + videoContainerId);
    selectedVideoContainer.style.display = 'block';

    // Update preview images
    var videoPreview1 = document.getElementById(prefix + 'video1Preview');
    var videoPreview2 = document.getElementById(prefix + 'video2Preview');

    videoPreview1.className = videoPreview1.className.replace(" preview-video-active", "");
    videoPreview2.className = videoPreview2.className.replace(" preview-video-active", "");

    document.getElementById(prefix + preview_id).className += " preview-video-active";
}


function isElementInViewport(el) {
    var rect = el.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

function isElementInViewport(el) {
    var rect = el.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

function checkVideoVisibility() {
    var videos = document.querySelectorAll('.auto-video');
    videos.forEach(function(video) {
        // Check if the video is in the viewport
        if (isElementInViewport(video)) {
            if (video.paused) {
                video.currentTime = 0; // Reset to start
                video.play();
            }
        } else {
            video.pause();
        }
    });
}

window.addEventListener('scroll', checkVideoVisibility);

document.addEventListener("DOMContentLoaded", function() {
    // Get all video elements with class 'video-music'
    var videos = document.querySelectorAll('.video-music');

    // Loop through each video and set the volume
    videos.forEach(function(video) {
        video.volume = 0.25; // 25% volume
    });
});


