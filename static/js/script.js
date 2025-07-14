  <script>
      document.addEventListener("DOMContentLoaded", function () {
        var clearBtn = document.getElementById("clear-result-btn");
        if (clearBtn) {
          clearBtn.addEventListener("click", function () {
            document.getElementById("status-text").textContent =
              "Awaiting result...";
            document.getElementById("progress-bar-span").style.width = "0%";
            document.getElementById("progress-bar-span").style.animation = "";
            document.getElementById("result-text").textContent =
              "No prediction yet";
          });
        }
        // Responsive menu toggle
        var menuBtn = document.getElementById("menu-btn");
        var mobileMenu = document.getElementById("mobile-menu");
        function setMobileMenuHeight() {
          if (window.innerWidth > 600) return;
          mobileMenu.style.height = mobileMenu.scrollHeight + "px";
        }
        function closeMobileMenu() {
          mobileMenu.style.height = mobileMenu.scrollHeight + "px";
          requestAnimationFrame(function () {
            mobileMenu.style.height = "0px";
          });
          setTimeout(function () {
            mobileMenu.style.display = "none";
          }, 350);
        }
        menuBtn.addEventListener("click", function () {
          if (
            mobileMenu.style.display === "none" ||
            mobileMenu.style.display === ""
          ) {
            mobileMenu.style.display = "flex";
            requestAnimationFrame(function () {
              setMobileMenuHeight();
            });
          } else {
            closeMobileMenu();
          }
        });
        // Hide mobile menu on resize if switching to desktop
        window.addEventListener("resize", function () {
          if (window.innerWidth > 600) {
            mobileMenu.style.display = "none";
            mobileMenu.style.height = "0px";
          }
        });
        // --- Mobile Simulate Dropdown Logic ---
        var simulateBtnMobile = document.getElementById("simulate-btn-mobile");
        var simulateDropdownMobile = document.getElementById(
          "simulate-dropdown-mobile"
        );
        var simulateDropdownWrapperMobile = document.getElementById(
          "simulate-dropdown-wrapper-mobile"
        );
        function openSimDropdown() {
          simulateDropdownWrapperMobile.classList.add("open");
          setMobileMenuHeight();
          simulateBtnMobile.setAttribute("aria-expanded", "true");
        }
        function closeSimDropdown() {
          simulateDropdownWrapperMobile.classList.remove("open");
          setMobileMenuHeight();
          simulateBtnMobile.setAttribute("aria-expanded", "false");
        }
        if (simulateBtnMobile && simulateDropdownMobile) {
          // Click to toggle
          simulateBtnMobile.addEventListener("click", function (e) {
            e.stopPropagation();
            var expanded =
              simulateBtnMobile.getAttribute("aria-expanded") === "true";
            if (!expanded) {
              openSimDropdown();
            } else {
              closeSimDropdown();
            }
          });
          // Hover to open/close (only on mobile view)
          simulateBtnMobile.addEventListener("mouseenter", function () {
            if (window.innerWidth <= 600) openSimDropdown();
          });
          simulateDropdownWrapperMobile.addEventListener(
            "mouseleave",
            function () {
              if (window.innerWidth <= 600) closeSimDropdown();
            }
          );
          // Hide dropdown when clicking outside
          document.addEventListener("click", function (e) {
            if (
              !simulateDropdownWrapperMobile.contains(e.target) &&
              e.target !== simulateBtnMobile &&
              window.innerWidth <= 600
            ) {
              closeSimDropdown();
            }
          });
          // Option actions
          var fill7Fields = window.fill7Fields;
          var fillAllFields = window.fillAllFields;
          var scrollToPredict = window.scrollToPredict;
          document.getElementById("simulate-7-mobile").onclick = function () {
            if (typeof fill7Fields === "function") fill7Fields();
            if (typeof scrollToPredict === "function") scrollToPredict();
            closeSimDropdown();
          };
          document.getElementById("simulate-all-mobile").onclick = function () {
            if (typeof fillAllFields === "function") fillAllFields();
            if (typeof scrollToPredict === "function") scrollToPredict();
            closeSimDropdown();
          };
        }
      });
      // Validation logic
      const form = document.querySelector("form");
      form.addEventListener("submit", function (e) {
        let valid = true;
        // Pregnancies
        const preg = document.getElementById("Pregnancies");
        if (preg.value === "" || preg.value < 0 || preg.value > 17) {
          document.getElementById("error-Pregnancies").textContent =
            "Enter 0–17";
          valid = false;
        } else {
          document.getElementById("error-Pregnancies").textContent = "";
        }
        // Family History
        const fam = document.getElementById("Family-history");
        if (fam.value === "") {
          document.getElementById("error-Family-history").textContent =
            "Select Yes or No";
          valid = false;
        } else {
          document.getElementById("error-Family-history").textContent = "";
        }
        // Physical Activity
        const pa = document.getElementById("Physical-Activity");
        if (pa.value === "") {
          document.getElementById("error-Physical-Activity").textContent =
            "Select activity";
          valid = false;
        } else {
          document.getElementById("error-Physical-Activity").textContent = "";
        }
        // Smoking Status
        const smoke = document.getElementById("Smoking-status");
        if (smoke.value === "") {
          document.getElementById("error-Smoking-status").textContent =
            "Select status";
          valid = false;
        } else {
          document.getElementById("error-Smoking-status").textContent = "";
        }
        // Alcohol Intake
        const alcohol = document.getElementById("Alcohol-Intake");
        if (alcohol.value === "" || alcohol.value < 0 || alcohol.value > 30) {
          document.getElementById("error-Alcohol-Intake").textContent =
            "0–30 drinks/week";
          valid = false;
        } else {
          document.getElementById("error-Alcohol-Intake").textContent = "";
        }
        // Diet Quality
        const diet = document.getElementById("Diet-Qualtiy");
        if (diet.value === "" || diet.value < 0 || diet.value > 10) {
          document.getElementById("error-Diet-Qualtiy").textContent =
            "0–10 servings/day";
          valid = false;
        } else {
          document.getElementById("error-Diet-Qualtiy").textContent = "";
        }
        // Cholesterol
        const chol = document.getElementById("Cholesterol");
        if (chol.value === "" || chol.value < 100 || chol.value > 400) {
          document.getElementById("error-Cholesterol").textContent =
            "100–400 mg/dL";
          valid = false;
        } else {
          document.getElementById("error-Cholesterol").textContent = "";
        }
        // Triglycerides
        const trig = document.getElementById("Triglycerides");
        if (trig.value === "" || trig.value < 50 || trig.value > 500) {
          document.getElementById("error-Triglycerides").textContent =
            "50–500 mg/dL";
          valid = false;
        } else {
          document.getElementById("error-Triglycerides").textContent = "";
        }
        // Waist Circumference
        const waist = document.getElementById("Waiste-ratio");
        if (waist.value === "" || waist.value < 60 || waist.value > 150) {
          document.getElementById("error-Waiste-ratio").textContent =
            "60–150 cm";
          valid = false;
        } else {
          document.getElementById("error-Waiste-ratio").textContent = "";
        }
        if (!valid) e.preventDefault();
      });
      // Simulate random input button logic
      // (Removed simulate-btn click handler to avoid conflict with dropdown)
      // Dropdown for simulate button (hover to show, closes on mouseleave/outside click)
      const simulateBtn = document.getElementById("simulate-btn");
      const dropdown = document.getElementById("simulate-dropdown");
      const simulateBtnWrapper = simulateBtn.parentElement;
      // Remove JS show/hide for dropdown, keep only click actions
      function fill7Fields() {
        document.querySelector('input[name="Age"]').value =
          Math.floor(Math.random() * 60) + 18;
        document.querySelector('input[name="BMI"]').value = (
          Math.random() * 15 +
          18
        ).toFixed(1);
        document.querySelector('input[name="Blood Glucose"]').value =
          Math.floor(Math.random() * 100) + 70;
        document.querySelector('input[name="Blood Pressure"]').value =
          Math.floor(Math.random() * 80) + 90;
        document.querySelector('input[name="HbA1c"]').value = (
          Math.random() * 6 +
          4
        ).toFixed(1);
        document.querySelector('input[name="Insulin Level"]').value =
          Math.floor(Math.random() * 100) + 20;
        document.querySelector('input[name="Skin thickness"]').value =
          Math.floor(Math.random() * 30) + 10;
        // Clear the rest
        document.getElementById("Pregnancies").value = "";
        document.getElementById("Family-history").value = "";
        document.getElementById("Physical-Activity").value = "";
        document.getElementById("Smoking-status").value = "";
        document.getElementById("Alcohol-Intake").value = "";
        document.getElementById("Diet-Qualtiy").value = "";
        document.getElementById("Cholesterol").value = "";
        document.getElementById("Triglycerides").value = "";
        document.getElementById("Waiste-ratio").value = "";
      }
      function fillAllFields() {
        fill7Fields();
        document.getElementById("Pregnancies").value = Math.floor(
          Math.random() * 10
        );
        document.getElementById("Family-history").value =
          Math.random() > 0.5 ? "1" : "0";
        document.getElementById("Physical-Activity").value = [
          "Low",
          "Medium",
          "High",
        ][Math.floor(Math.random() * 3)];
        document.getElementById("Smoking-status").value =
          Math.random() > 0.5 ? "Smoker" : "Non-Smoker";
        document.getElementById("Alcohol-Intake").value = Math.floor(
          Math.random() * 31
        );
        document.getElementById("Diet-Qualtiy").value = Math.floor(
          Math.random() * 11
        );
        document.getElementById("Cholesterol").value =
          Math.floor(Math.random() * 301) + 100;
        document.getElementById("Triglycerides").value =
          Math.floor(Math.random() * 451) + 50;
        document.getElementById("Waiste-ratio").value =
          Math.floor(Math.random() * 91) + 60;
      }
      function scrollToPredict() {
        var predictBtn = document.querySelector('form button[type="submit"]');
        if (predictBtn) {
          predictBtn.scrollIntoView({ behavior: "smooth", block: "center" });
          predictBtn.focus();
        }
      }
      if (document.getElementById("simulate-7")) {
        document.getElementById("simulate-7").onclick = function () {
          fill7Fields();
          scrollToPredict();
        };
      }
      if (document.getElementById("simulate-all")) {
        document.getElementById("simulate-all").onclick = function () {
          fillAllFields();
          scrollToPredict();
        };
      }
      // CSS fix for dropdown z-index and overflow
      const style = document.createElement("style");
      style.innerHTML = `
        #simulate-dropdown { z-index: 9999 !important; }
        .header-main, .header-actions { overflow: visible !important; }
      `;
      document.head.appendChild(style);
    </script>