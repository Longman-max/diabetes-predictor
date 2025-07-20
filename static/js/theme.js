// Theme switching functionality
function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute("data-theme");
  const newTheme = currentTheme === "dark" ? "light" : "dark";

  document.documentElement.setAttribute("data-theme", newTheme);
  localStorage.setItem("theme", newTheme);

  // Update the icon
  const themeIcon = document.querySelector(".theme-btn span");
  themeIcon.textContent = newTheme === "dark" ? "dark_mode" : "light_mode";
}

// Initialize theme on page load
document.addEventListener("DOMContentLoaded", () => {
  const savedTheme = localStorage.getItem("theme") || "light";
  document.documentElement.setAttribute("data-theme", savedTheme);

  const themeIcon = document.querySelector(".theme-btn span");
  if (themeIcon) {
    themeIcon.textContent = savedTheme === "dark" ? "dark_mode" : "light_mode";
  }

  // Set up event listener for theme toggle button
  const themeBtn = document.querySelector(".theme-btn");
  if (themeBtn) {
    themeBtn.addEventListener("click", toggleTheme);
  }
});
