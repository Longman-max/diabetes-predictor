    document.addEventListener("DOMContentLoaded", function () {
  var clearBtn = document.getElementById("clear-result-btn");
  if (clearBtn) {
    clearBtn.addEventListener("click", function () {
      document.getElementById("status-text").textContent = "Awaiting result...";
      document.getElementById("progress-bar-span").style.width = "0%";
      document.getElementById("progress-bar-span").style.animation = "";
      document.getElementById("result-text").textContent = "No prediction yet";
      // updateHealthIcon("none"); // Reset icon
    });
  }

  // Function to update health icon based on risk level
  function updateHealthIcon(risk) {
    const healthIcon = document.querySelector('.health-icon');
    if (!healthIcon) return;

    // Remove any existing risk classes
    healthIcon.classList.remove('high-risk', 'medium-risk', 'low-risk');

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

  // Check prediction result when available
  const resultText = document.getElementById('result-text');
  if (resultText && resultText.textContent !== "No prediction yet") {
    const text = resultText.textContent.toLowerCase();
    if (text.includes('high risk') || text.includes('high likelihood')) {
      updateHealthIcon('high');
    } else if (text.includes('medium risk') || text.includes('moderate')) {
      updateHealthIcon('medium');
    } else if (text.includes('low risk') || text.includes('unlikely')) {
      updateHealthIcon('low');
    }
  }
});
// Validation logic
const form = document.querySelector("form");
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
  const waist = document.getElementById("Waiste-ratio");
  if (waist.value === "" || waist.value < 60 || waist.value > 150) {
    document.getElementById("error-Waiste-ratio").textContent = "60–150 cm";
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
function fillFields(riskLevel) {
  // Helper function to get random number within range
  const getRandomInRange = (min, max, decimals = 0) => {
    const num = Math.random() * (max - min) + min;
    return decimals ? Number(num.toFixed(decimals)) : Math.floor(num);
  };

  // Clinical reference ranges for all fields
  let ranges = {
    Age: [18, 80],
    BMI: [18.5, 40, 1],
    'Blood Glucose': [70, 200],
    'Blood Pressure': [80, 180],
    'HbA1c': [4.0, 12.0, 1],
    'Insulin Level': [2, 300],
    'Skin thickness': [7, 99],
    Pregnancies: [0, 17],
    'Family history': [0, 1],
    'Physical Activity': ["Low", "Medium", "High"],
    'Smoking status': ["Smoker", "Non-Smoker"],
    'Alcohol Intake': [0, 30],
    'Diet Qualtiy': [0, 10],
    Cholesterol: [100, 400],
    Triglycerides: [50, 500],
    'Waiste ratio': [60, 150]
  };

  let values = {
    Age: getRandomInRange(...ranges.Age),
    BMI: getRandomInRange(...ranges.BMI),
    'Blood Glucose': getRandomInRange(...ranges['Blood Glucose']),
    'Blood Pressure': getRandomInRange(...ranges['Blood Pressure']),
    'HbA1c': getRandomInRange(...ranges['HbA1c']),
    'Insulin Level': getRandomInRange(...ranges['Insulin Level']),
    'Skin thickness': getRandomInRange(...ranges['Skin thickness']),
    Pregnancies: getRandomInRange(...ranges.Pregnancies),
    'Family history': String(getRandomInRange(...ranges['Family history'])),
    'Physical Activity': ranges['Physical Activity'][getRandomInRange(0, 2)],
    'Smoking status': ranges['Smoking status'][getRandomInRange(0, 1)],
    'Alcohol Intake': getRandomInRange(...ranges['Alcohol Intake']),
    'Diet Qualtiy': getRandomInRange(...ranges['Diet Qualtiy']),
    Cholesterol: getRandomInRange(...ranges.Cholesterol),
    Triglycerides: getRandomInRange(...ranges.Triglycerides),
    'Waiste ratio': getRandomInRange(...ranges['Waiste ratio'])
  };

  // Fill in the form fields
  Object.entries(values).forEach(([key, value]) => {
    const element = document.querySelector(`input[name="${key}"]`) || document.getElementById(`${key.replace(/\s+/g, '-')}`);
    if (element) element.value = value;
    // For select fields
    if (key === 'Family history') {
      const select = document.getElementById('Family-history');
      if (select) select.value = value;
    }
    if (key === 'Physical Activity') {
      const select = document.getElementById('Physical-Activity');
      if (select) select.value = value;
    }
    if (key === 'Smoking status') {
      const select = document.getElementById('Smoking-status');
      if (select) select.value = value;
    }
  });
}

function scrollToPredict() {
  var predictBtn = document.querySelector('form button[type="submit"]');
  if (predictBtn) {
    predictBtn.scrollIntoView({ behavior: "smooth", block: "center" });
    predictBtn.focus();
  }
}

// Set up simulation button handlers
document.getElementById("simulate-btn").onclick = () => {
  fillFields('random');
  scrollToPredict();
};
document.getElementById("simulate-btn-mobile").onclick = () => {
  fillFields('random');
  scrollToPredict();
  document.getElementById('mobile-menu').classList.remove('show');
};
// CSS fix for dropdown z-index and overflow
const style = document.createElement("style");
style.innerHTML = `
        #simulate-dropdown { z-index: 9999 !important; }
        .header-main, .header-actions { overflow: visible !important; }
      `;
document.head.appendChild(style);      // Mobile menu functionality
    const menuBtn = document.getElementById('menu-btn');
    const mobileMenu = document.getElementById('mobile-menu');

    if (menuBtn && mobileMenu) {
      menuBtn.addEventListener('click', () => {
        mobileMenu.classList.toggle('show');
      });

      // Close menu when clicking outside
      document.addEventListener('click', (e) => {
        if (!mobileMenu.contains(e.target) && !menuBtn.contains(e.target)) {
          mobileMenu.classList.remove('show');
        }
      });
    }