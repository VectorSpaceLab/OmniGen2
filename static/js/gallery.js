// const galleries = [
//     { id: 'text-to-image', title: 'Text-to-Image Generation' },
//     { id: 'instruction-tuning', title: 'Instruction Tuning' },
//     { id: 'reasoning', title: 'Reasoning and Knowledge-Augmented Generation' },
//     { id: 'image-editing', title: 'Image Editing' },
// ];
const galleries = [
    { id: 'text-to-image', title: 'Text-to-Image Generation' },
    { id: 'image-editing', title: 'Image Editing' },
    { id: 'subject-driven-generation', title: 'In-context Generation' },
];

let currentGalleryIndex = 0;

function updateGallery() {
    // Hide all galleries
    document.querySelectorAll('.gallery').forEach(gallery => {
        gallery.classList.remove('active');
    });

    // Show current gallery
    const currentGallery = document.getElementById(galleries[currentGalleryIndex].id);
    currentGallery.classList.add('active');

    // Update title
    document.querySelector('.gallery-title').textContent = galleries[currentGalleryIndex].title;

    // Update button states
    const prevBtn = document.querySelector('.prev-btn');
    const nextBtn = document.querySelector('.next-btn');
    
    prevBtn.disabled = currentGalleryIndex === 0;
    nextBtn.disabled = currentGalleryIndex === galleries.length - 1;
}

function nextGallery() {
    if (currentGalleryIndex < galleries.length - 1) {
        currentGalleryIndex++;
        updateGallery();
    }
}

function prevGallery() {
    if (currentGalleryIndex > 0) {
        currentGalleryIndex--;
        updateGallery();
    }
}

// Image Comparison Slider Functionality
function initImageComparisons() {
    const comparisons = document.querySelectorAll('.image-comparison');
    
    comparisons.forEach(comparison => {
        const slider = comparison.querySelector('.slider');
        const originalImage = comparison.querySelector('.original-image');
        let isDragging = false;
        let startX = 0;
        let sliderLeft = 0;

        // Set initial position to 50%
        slider.style.left = '50%';
        originalImage.style.clipPath = 'polygon(0 0, 50% 0, 50% 100%, 0 100%)';

        // Mouse events
        slider.addEventListener('mousedown', startDragging);
        document.addEventListener('mouseup', stopDragging);
        document.addEventListener('mousemove', drag);

        // Touch events
        slider.addEventListener('touchstart', handleTouchStart);
        document.addEventListener('touchend', handleTouchEnd);
        document.addEventListener('touchmove', handleTouchMove);

        function startDragging(e) {
            isDragging = true;
            startX = e.clientX - slider.offsetLeft;
            e.preventDefault();
        }

        function handleTouchStart(e) {
            isDragging = true;
            startX = e.touches[0].clientX - slider.offsetLeft;
            e.preventDefault();
        }

        function stopDragging() {
            isDragging = false;
        }

        function handleTouchEnd() {
            isDragging = false;
        }

        function drag(e) {
            if (!isDragging) return;
            
            const rect = comparison.getBoundingClientRect();
            const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
            updateSliderPosition(x, rect.width);
        }

        function handleTouchMove(e) {
            if (!isDragging) return;
            
            const rect = comparison.getBoundingClientRect();
            const touch = e.touches[0];
            const x = Math.max(0, Math.min(touch.clientX - rect.left, rect.width));
            updateSliderPosition(x, rect.width);
            e.preventDefault();
        }

        function updateSliderPosition(x, width) {
            const percentage = (x / width) * 100;
            slider.style.left = `${percentage}%`;
            originalImage.style.clipPath = `polygon(0 0, ${percentage}% 0, ${percentage}% 100%, 0 100%)`;
        }
    });
}

function initImageSwitches() {
    const containers = document.querySelectorAll('.image-switch-container');
    
    containers.forEach(container => {
        const button = container.querySelector('.generate-btn');
        
        button.addEventListener('click', () => {
            container.classList.toggle('generated');
            button.textContent = container.classList.contains('generated') ? 'Original' : 'Generate';
        });
    });
}

// Initialize gallery and image interactions when the page loads
document.addEventListener('DOMContentLoaded', () => {
    updateGallery();
    initImageComparisons();
    initImageSwitches();
}); 