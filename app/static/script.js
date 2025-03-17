// Particle Background
particlesJS('particles-js', {
    particles: {
        number: { value: 60, density: { enable: true, value_area: 1000 } },
        color: { value: '#ff9500' },
        shape: { type: 'circle' },
        opacity: { value: 0.8, random: true },
        size: { value: 4, random: true },
        move: { enable: true, speed: 1.5, direction: 'none', random: true }
    },
    interactivity: {
        events: { onhover: { enable: true, mode: 'repulse' }, onclick: { enable: true, mode: 'push' } },
        modes: { repulse: { distance: 100 }, push: { particles_nb: 4 } }
    }
});

// Demo Buttons
document.querySelectorAll('.demo-btn').forEach(button => {
    button.addEventListener('click', function() {
        const text = this.getAttribute('data-text');
        document.getElementById('user_text').value = text;
        updateCharCount();
    });
});

// Character Count for Textarea
const textarea = document.getElementById('user_text');
const charCount = document.getElementById('charCount');

function updateCharCount() {
    const count = textarea.value.length;
    charCount.textContent = `${count}/160`;
    textarea.style.boxShadow = count > 0 ? '0 0 15px rgba(255, 149, 0, 0.6)' : 'none';
}

textarea.addEventListener('input', updateCharCount);

// Theme Toggle with Persistence
const themeToggle = document.getElementById('themeToggle');
const savedTheme = localStorage.getItem('theme');
if (savedTheme === 'void') {
    document.body.classList.add('void-theme');
}
themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('void-theme');
    localStorage.setItem('theme', document.body.classList.contains('void-theme') ? 'void' : 'default');
});

// Scroll to Top
const scrollTopBtn = document.getElementById('scrollTop');
window.addEventListener('scroll', () => {
    scrollTopBtn.style.display = window.scrollY > 300 ? 'block' : 'none';
});
scrollTopBtn.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
});

// Tooltips for Progress Bar
const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
tooltipTriggerList.forEach(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));

// 3D Tilt Effect for Cards
document.querySelectorAll('.card').forEach(card => {
    card.addEventListener('mousemove', (e) => {
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        const tiltX = (y - centerY) / 20;
        const tiltY = (centerX - x) / 20;
        card.style.transform = `perspective(1000px) rotateX(${tiltX}deg) rotateY(${tiltY}deg) translateY(-5px)`;
    });
    card.addEventListener('mouseleave', () => {
        card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) translateY(0)';
    });
});

// Feedback Form Submission Animation
const feedbackForm = document.getElementById('feedbackForm');
if (feedbackForm) {
    feedbackForm.addEventListener('submit', function(e) {
        const submitBtn = this.querySelector('.btn-primary');
        submitBtn.textContent = 'Submitting...';
        submitBtn.disabled = true;
        submitBtn.style.background = 'linear-gradient(45deg, #ffaa33, #ff9500)';
        setTimeout(() => {
            submitBtn.textContent = 'Submitted!';
            submitBtn.style.background = '#ff8000';
            setTimeout(() => {
                submitBtn.textContent = 'Submit Feedback';
                submitBtn.disabled = false;
                submitBtn.style.background = 'linear-gradient(45deg, #ff9500, #ffaa33)';
            }, 1000);
        }, 1000);
    });
}