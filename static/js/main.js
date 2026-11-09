document.addEventListener("DOMContentLoaded", () => {
    // Theme Toggle
    const themeBtn = document.getElementById("themeToggleBtn");
    const themeIcon = document.getElementById("themeIcon");
    const htmlElem = document.documentElement;

    const savedTheme = localStorage.getItem("tg_theme") || "light";
    htmlElem.setAttribute("data-bs-theme", savedTheme);
    updateThemeIcon(savedTheme);

    if (themeBtn) {
        themeBtn.addEventListener("click", () => {
            const current = htmlElem.getAttribute("data-bs-theme");
            const next = current === "dark" ? "light" : "dark";
            htmlElem.setAttribute("data-bs-theme", next);
            localStorage.setItem("tg_theme", next);
            updateThemeIcon(next);
        });
    }

    function updateThemeIcon(theme) {
        if (!themeIcon) return;
        themeIcon.className = theme === "dark" ? "fa-solid fa-sun text-warning" : "fa-solid fa-moon";
    }

    // Scroll to Top
    const scrollTopBtn = document.getElementById("scrollTopBtn");
    window.addEventListener("scroll", () => {
        if (scrollTopBtn) {
            if (window.scrollY > 300) {
                scrollTopBtn.classList.remove("d-none");
            } else {
                scrollTopBtn.classList.add("d-none");
            }
        }
    });

    if (scrollTopBtn) {
        scrollTopBtn.addEventListener("click", () => {
            window.scrollTo({ top: 0, behavior: "smooth" });
        });
    }

    // Tagline Typing Effect
    const taglineElem = document.getElementById("typingTagline");
    if (taglineElem) {
        const text = "Plan Smarter. Travel Better.";
        let index = 0;
        function typeText() {
            if (index < text.length) {
                taglineElem.textContent += text.charAt(index);
                index++;
                setTimeout(typeText, 90);
            }
        }
        taglineElem.textContent = "";
        typeText();
    }
});

function showToast(message, isError = false) {
    const toastEl = document.getElementById('appToast');
    const toastBody = document.getElementById('toastMessage');
    if (toastEl && toastBody) {
        toastBody.innerText = message;
        toastEl.className = `toast align-items-center text-white border-0 ${isError ? 'bg-danger' : 'bg-success'}`;
        const toast = new bootstrap.Toast(toastEl);
        toast.show();
    }
}
