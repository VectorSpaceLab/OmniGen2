document.addEventListener("scroll", function() {
    let navLinks = document.querySelectorAll('.nav-bar .nav-link');

    let navBar = document.getElementById('nav-bar');
    let header = document.getElementById('header-container');

    // Check the scroll position\
    headerBottom = header.getBoundingClientRect().bottom;
    if (headerBottom > 0) {
        navBar.style.display = 'none';
    } else {
        navBar.style.display = 'block';
    }

    // check for mobile window size
    if (window.innerWidth < 768) {
        navBar.style.display = 'none';
    }

    navLinks.forEach(link => {
        let section = document.querySelector(link.getAttribute('href'));
        if (!section) return; // Skip if the section doesn't exist

        const sectionTop = section.getBoundingClientRect().top;
        const sectionBottom = section.getBoundingClientRect().bottom;

        // Check if the section is in the viewport
        if (sectionTop < window.innerHeight && sectionBottom >= 0) {
            link.classList.add('active-link');
        } else {
            link.classList.remove('active-link');
        }
    });
});
