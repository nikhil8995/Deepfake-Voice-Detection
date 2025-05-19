// --- Dark/Light Mode Switch ---
const themeToggle = document.getElementById('theme-toggle');
const themeIcon = document.getElementById('theme-icon');
const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

function setTheme(dark) {
    if (dark) {
        document.body.classList.add('dark-mode');
        themeIcon.classList.remove('bi-moon');
        themeIcon.classList.add('bi-sun');
        themeToggle.setAttribute('aria-label', 'Switch to light mode');
        localStorage.setItem('theme', 'dark');
    } else {
        document.body.classList.remove('dark-mode');
        themeIcon.classList.remove('bi-sun');
        themeIcon.classList.add('bi-moon');
        themeToggle.setAttribute('aria-label', 'Switch to dark mode');
        localStorage.setItem('theme', 'light');
    }
}
if (themeToggle && themeIcon) {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
        setTheme(true);
    } else {
        setTheme(false);
    }
    themeToggle.addEventListener('click', () => {
        setTheme(!document.body.classList.contains('dark-mode'));
    });
}

// Loading overlay
document.getElementById('uploadForm').onsubmit = function() {
    document.getElementById('loading-overlay').style.display = 'flex';
};

// Drag & drop file input
const dropArea = document.getElementById('drop-area');
const fileInput = dropArea.querySelector('input[type="file"]');
const labelText = document.getElementById('file-label-text');

dropArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropArea.classList.add('dragover');
});
dropArea.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropArea.classList.remove('dragover');
});
dropArea.addEventListener('drop', (e) => {
    e.preventDefault();
    dropArea.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        labelText.textContent = e.dataTransfer.files[0].name;
        showWaveform(fileInput.files[0]);
    }
});
fileInput.addEventListener('change', (e) => {
    if (fileInput.files.length) {
        labelText.textContent = fileInput.files[0].name;
        showWaveform(fileInput.files[0]);
    }
});

// Animate Predict button on click (ripple effect)
document.querySelector('.btn-primary').addEventListener('click', function(e) {
    const btn = this;
    const ripple = document.createElement('span');
    ripple.className = 'ripple';
    ripple.style.left = (e.offsetX || e.touches?.[0]?.clientX || 0) + 'px';
    ripple.style.top = (e.offsetY || e.touches?.[0]?.clientY || 0) + 'px';
    btn.appendChild(ripple);
    setTimeout(() => ripple.remove(), 600);
});

// Waveform preview using WaveSurfer.js
let wavesurfer = null;
function showWaveform(file) {
    const container = document.getElementById('waveform-container');
    const waveformDiv = document.getElementById('waveform');
    container.style.display = 'block';
    waveformDiv.innerHTML = '';
    if (wavesurfer) {
        wavesurfer.destroy();
    }
    wavesurfer = WaveSurfer.create({
        container: waveformDiv,
        waveColor: '#6366f1',
        progressColor: '#60a5fa',
        height: 60,
        barWidth: 2,
        barRadius: 2,
        responsive: true,
        cursorColor: '#3b82f6'
    });
    const reader = new FileReader();
    reader.onload = function(e) {
        wavesurfer.loadBlob(new Blob([e.target.result]));
    };
    reader.readAsArrayBuffer(file);
}

// Confetti animation for "Real" result
window.addEventListener('DOMContentLoaded', () => {
    const resultBox = document.getElementById('result-box');
    if (resultBox && resultBox.classList.contains('result-real')) {
        launchConfetti();
    }
});

function launchConfetti() {
    const canvas = document.getElementById('confetti-canvas');
    canvas.style.display = 'block';
    const ctx = canvas.getContext('2d');
    let W = window.innerWidth, H = window.innerHeight;
    canvas.width = W; canvas.height = H;
    let particles = [];
    for (let i = 0; i < 120; i++) {
        particles.push({
            x: Math.random() * W,
            y: Math.random() * H - H,
            r: Math.random() * 6 + 4,
            d: Math.random() * 100,
            color: `hsl(${Math.random()*360},90%,60%)`,
            tilt: Math.random() * 10 - 10
        });
    }
    let angle = 0;
    function draw() {
        ctx.clearRect(0, 0, W, H);
        angle += 0.01;
        for (let i = 0; i < particles.length; i++) {
            let p = particles[i];
            ctx.beginPath();
            ctx.fillStyle = p.color;
            ctx.ellipse(p.x, p.y, p.r, p.r/2, p.tilt, 0, 2 * Math.PI);
            ctx.fill();
            p.y += (Math.cos(angle + p.d) + 1 + p.r/2) * 1.2;
            p.x += Math.sin(angle) * 2;
            if (p.y > H) {
                p.x = Math.random() * W;
                p.y = -10;
            }
        }
        requestAnimationFrame(draw);
    }
    draw();
    setTimeout(() => { canvas.style.display = 'none'; }, 3500);
}

// Responsive confetti canvas
window.addEventListener('resize', () => {
    const canvas = document.getElementById('confetti-canvas');
    if (canvas) {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
});