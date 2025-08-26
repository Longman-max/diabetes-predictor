// Define the SVG based on risk level
const medicalBriefcaseSVG = `<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-briefcase-medical-icon lucide-briefcase-medical">
            <path d="M12 11v4"/>
            <path d="M14 13h-4"/>
            <path d="M16 6V4a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v2"/>
            <path d="M18 6v14"/>
            <path d="M6 6v14"/>
            <rect width="20" height="14" x="2" y="6" rx="2"/>
          </svg>`;

const alertTriangleSVG = `<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-triangle-alert-icon lucide-triangle-alert">
            <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>
            <path d="M12 9v4"/>
            <path d="M12 17h.01"/>
          </svg>`;


document.addEventListener("DOMContentLoaded", function () {
  var clearBtn = document.getElementById("clear-result-btn");
  if (clearBtn) {
    clearBtn.addEventListener("click", function () {
      document.getElementById("status-text").textContent = "Awaiting result...";
      document.getElementById("progress-bar-span").style.width = "0%";
      document.getElementById("progress-bar-span").style.animation = "";
      document.getElementById("result-text").textContent = "No prediction yet";
      document.getElementById("advice-text").textContent = "";
      document.querySelector('.health-icon').innerHTML = medicalBriefcaseSVG;
      document.querySelector('.health-icon').classList.add('default-icon');
    });
  }

  // Function to update health icon based on risk level
  function updateHealthIcon(risk) {
    const healthIcon = document.querySelector('.health-icon');
    if (!healthIcon) return;

    // Remove any existing risk classes
    healthIcon.classList.remove('high-risk', 'medium-risk', 'low-risk');

    // Update icon and class based on risk level
    if (risk === "high") {
      healthIcon.innerHTML = alertTriangleSVG;
      healthIcon.classList.add('high-risk');
    } else if (risk === "medium") {
      healthIcon.innerHTML = alertTriangleSVG;
      healthIcon.classList.add('medium-risk');
    } else {
      healthIcon.innerHTML = medicalBriefcaseSVG;
      healthIcon.classList.add('low-risk');
    }
  }

  // Check prediction result when available (e.g., if page reloads with result)
  const resultText = document.getElementById('result-text');
  if (resultText && resultText.textContent !== "No prediction yet") {
    const text = resultText.textContent.toLowerCase();
    if (text.includes('high risk') || text.includes('high likelihood')) {
      updateHealthIcon('high');
    } else if (text.includes('medium risk') || text.includes('moderate')) {
      updateHealthIcon('medium');
    } else if (text.includes('no risk') || text.includes('unlikely')) {
      updateHealthIcon('low');
    }
  }

    // Set up simulation button handlers
    const simulateBtn = document.getElementById("simulate-btn");
    if(simulateBtn) {
        simulateBtn.addEventListener('click', () => {
            fillFields();
            scrollToPredict();
        });
    }

    const simulateBtnMobile = document.getElementById("simulate-btn-mobile");
    if(simulateBtnMobile) {
        simulateBtnMobile.addEventListener('click', () => {
            fillFields();
            scrollToPredict();
            // Optional: close mobile menu if it exists
            const mobileMenu = document.getElementById('mobile-menu');
            if (mobileMenu) {
                mobileMenu.classList.remove('show');
            }
        });
    }
});

// Validation logic
const form = document.querySelector("form");
if(form) {
    form.addEventListener("submit", function (e) {
      let valid = true;
      // Pregnancies
      const preg = document.getElementById("Pregnancies");
      if (preg.value === "" || preg.value < 0 || preg.value > 17) {
        document.getElementById("error-Pregnancies").textContent = "Enter 0–17";
        valid = false;
      } else {
        document.getElementById("error-Pregnancies").textContent = "";
      }
      // Family History
      const fam = document.getElementById("Family-history");
      if (fam.value === "") {
        document.getElementById("error-Family-history").textContent = "Select Yes or No";
        valid = false;
      } else {
        document.getElementById("error-Family-history").textContent = "";
      }
      // Physical Activity
      const pa = document.getElementById("Physical-Activity");
      if (pa.value === "") {
        document.getElementById("error-Physical-Activity").textContent = "Select activity";
        valid = false;
      } else {
        document.getElementById("error-Physical-Activity").textContent = "";
      }
      // Smoking Status
      const smoke = document.getElementById("Smoking-status");
      if (smoke.value === "") {
        document.getElementById("error-Smoking-status").textContent = "Select status";
        valid = false;
      } else {
        document.getElementById("error-Smoking-status").textContent = "";
      }
      // Alcohol Intake
      const alcohol = document.getElementById("Alcohol-Intake");
      if (alcohol.value === "" || alcohol.value < 0 || alcohol.value > 30) {
        document.getElementById("error-Alcohol-Intake").textContent = "0–30 drinks/week";
        valid = false;
      } else {
        document.getElementById("error-Alcohol-Intake").textContent = "";
      }
      // Diet Type
      const diet = document.getElementById("Diet_Type");
      if (diet.value === "") {
        document.getElementById("error-Diet_Type").textContent = "Select a diet type";
        valid = false;
      } else {
        document.getElementById("error-Diet_Type").textContent = "";
      }
      // Cholesterol
      const chol = document.getElementById("Cholesterol");
      if (chol.value === "" || chol.value < 100 || chol.value > 400) {
        document.getElementById("error-Cholesterol").textContent = "100–400 mg/dL";
        valid = false;
      } else {
        document.getElementById("error-Cholesterol").textContent = "";
      }
      // Triglycerides
      const trig = document.getElementById("Triglycerides");
      if (trig.value === "" || trig.value < 50 || trig.value > 500) {
        document.getElementById("error-Triglycerides").textContent = "50–500 mg/dL";
        valid = false;
      } else {
        document.getElementById("error-Triglycerides").textContent = "";
      }
      // Waist Circumference
      const waist = document.getElementById("Waist-ratio");
      if (waist.value === "" || waist.value < 60 || waist.value > 150) {
        document.getElementById("error-Waist-ratio").textContent = "60–150 cm";
        valid = false;
      } else {
        document.getElementById("error-Waist-ratio").textContent = "";
      }
      if (!valid) e.preventDefault();
    });
}

/**
 * Fills the form with random data.
 */
function fillFields() {
  // Helper function to get a random number within a range
  const getRandomInRange = (min, max, decimals = 0) => {
    const num = Math.random() * (max - min) + min;
    return decimals ? Number(num.toFixed(decimals)) : Math.floor(num);
  };

  // Helper function to get a random item from an array
  const getRandomOption = (options) => {
    const index = getRandomInRange(0, options.length);
    return options[index];
  };

  // Define the ranges and options for each form field
  const fieldData = {
    'Age': () => getRandomInRange(18, 80),
    'BMI': () => getRandomInRange(18.5, 40, 1),
    'Blood Glucose': () => getRandomInRange(70, 200),
    'Blood Pressure': () => getRandomInRange(80, 180),
    'HbA1c': () => getRandomInRange(4.0, 12.0, 1),
    'Insulin Level': () => getRandomInRange(2, 300),
    'Skin thickness': () => getRandomInRange(7, 99),
    'Pregnancies': () => getRandomInRange(0, 17),
    'Family history': () => getRandomOption(['0', '1']),
    'Physical Activity': () => getRandomOption(["Low", "Medium", "High"]),
    'Smoking status': () => getRandomOption(["Smoker", "Non-Smoker"]),
    'Alcohol Intake': () => getRandomInRange(0, 30),
    'Diet_Type': () => getRandomOption(["Non-Vegetarian", "Vegetarian", "Vegan"]),
    'Cholesterol': () => getRandomInRange(100, 400),
    'Triglycerides': () => getRandomInRange(50, 500),
    'Waist ratio': () => getRandomInRange(60, 150)
  };

  // Loop through the fields and populate them
  for (const [name, valueFn] of Object.entries(fieldData)) {
    // This robust selector finds any form element with the matching 'name' attribute
    const element = document.querySelector(`[name="${name}"]`);
    if (element) {
      element.value = valueFn();
    } else {
      console.warn(`Simulate fields: Element with name "${name}" not found.`);
    }
  }
}

/**
 * Smoothly scrolls to the "Predict" button.
 */
function scrollToPredict() {
  const predictBtn = document.querySelector('form button[type="submit"]');
  if (predictBtn) {
    predictBtn.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}