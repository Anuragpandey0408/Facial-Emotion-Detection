function toggleMenu() {
    const navLinks = document.querySelector('.nav-links');
    navLinks.classList.toggle('active'); // Toggle the active class
}

// Close the menu when clicking outside
window.onclick = function (event) {
    if (!event.target.matches('.hamburger') && !event.target.matches('.nav-links') && !event.target.matches('.nav-links *')) {
        const navLinks = document.querySelector('.nav-links');
        if (navLinks.classList.contains('active')) {
            navLinks.classList.remove('active'); // Remove active class
        }
    }
} 