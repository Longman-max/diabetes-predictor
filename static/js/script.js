// --- SVG icons based on risk level ---
const medicalBriefcaseSVG = `<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-briefcase-medical">
    <path d="M12 11v4"/><path d="M14 13h-4"/><path d="M16 6V4a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v2"/><path d="M18 6v14"/><path d="M6 6v14"/><rect width="20" height="14" x="2" y="6" rx="2"/></svg>`;

const alertTriangleSVG = `<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-triangle-alert">
    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>`;

const checkCircleSVG = `<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-check-circle">
    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/></svg>`;

// --- Function to update health icon ---
function updateHealthIcon(className) {
  const healthIcon = document.querySelector(".health-icon");
  if (!healthIcon) return;

  healthIcon.classList.remove("high-risk", "moderate-risk", "fair-risk", "non-diabetic", "default-icon");
  healthIcon.classList.add(className);

  if (className === "high-risk" || className === "moderate-risk" || className === "fair-risk") {
    healthIcon.innerHTML = alertTriangleSVG;
  } else if (className === "non-diabetic") {
    healthIcon.innerHTML = checkCircleSVG;
  } else {
    healthIcon.innerHTML = medicalBriefcaseSVG;
  }
}

// --- Function to display results ---
function displayPredictionResult(prediction, advice, className) {
  console.log('Displaying result:', { prediction, advice, className });
  
  document.getElementById("status-text").textContent = "Prediction complete";
  document.getElementById("progress-bar-span").style.width = "100%";
  document.getElementById("result-text").textContent = prediction;
  document.getElementById("advice-text").textContent = advice;

  const resultBox = document.getElementById("result-box");
  resultBox.className = "prediction-box"; // reset
  resultBox.classList.add(className);

  updateHealthIcon(className);
}

// --- Validation function ---
function validateForm(formData) {
  const errors = [];
  
  // Required fields validation
  const requiredFields = [
    'Age', 'BMI', 'Blood Glucose', 'Blood Pressure', 'HbA1c',
    'Insulin Level', 'Skin thickness', 'Pregnancies', 'Family history',
    'Physical Activity', 'Smoking status', 'Alcohol Intake', 'Diet_Type',
    'Cholesterol', 'Triglycerides', 'Waist ratio'
  ];
  
  requiredFields.forEach(field => {
    const value = formData.get(field);
    if (!value || value.trim() === '') {
      errors.push(`${field} is required`);
    }
  });
  
  // Numeric range validations
  const age = parseFloat(formData.get('Age'));
  if (age < 10 || age > 100) {
    errors.push('Age must be between 10-100');
  }
  
  const bmi = parseFloat(formData.get('BMI'));
  if (bmi < 10 || bmi > 60) {
    errors.push('BMI must be between 10-60');
  }
  
  const glucose = parseFloat(formData.get('Blood Glucose'));
  if (glucose < 50 || glucose > 800) {
    errors.push('Blood Glucose must be between 50-800');
  }
  
  const bp = parseFloat(formData.get('Blood Pressure'));
  if (bp < 80 || bp > 200) {
    errors.push('Blood Pressure must be between 80-200');
  }
  
  const hba1c = parseFloat(formData.get('HbA1c'));
  if (hba1c < 4.0 || hba1c > 14.0) {
    errors.push('HbA1c must be between 4.0-14.0');
  }
  
  return errors;
}

// --- DOM Content Loaded ---
document.addEventListener("DOMContentLoaded", function () {
  const predictionForm = document.getElementById("prediction-form");
  const clearBtn = document.getElementById("clear-result-btn");

  // --- Handle form submission with Fetch API ---
  if (predictionForm) {
    predictionForm.addEventListener("submit", function (event) {
      event.preventDefault();

      const formData = new FormData(predictionForm);
      
      // Validate form
      const validationErrors = validateForm(formData);
      if (validationErrors.length > 0) {
        alert('Please fix the following errors:\n' + validationErrors.join('\n'));
        return;
      }
      
      console.log('Submitting form data:', Object.fromEntries(formData.entries()));

      // Update UI to show prediction in progress
      document.getElementById("status-text").textContent = "Predicting...";
      document.getElementById("progress-bar-span").style.width = "50%";
      document.getElementById("result-text").textContent = "";
      document.getElementById("advice-text").textContent = "";
      document.getElementById("result-box").className = "prediction-box";

      fetch(predictionForm.action, {
        method: predictionForm.method,
        body: formData,
      })
        .then((response) => {
          console.log('Response status:', response.status);
          if (!response.ok) {
            return response.json().then(err => Promise.reject(err));
          }
          return response.json();
        })
        .then((data) => {
          console.log('Received data:', data);
          if (data.error) {
            throw new Error(data.error);
          }
          displayPredictionResult(data.message, data.advice, data.class_name);
        })
        .catch((error) => {
          console.error("Error:", error);
          document.getElementById("result-text").textContent = "Prediction failed: " + (error.message || error.error || "Unknown error");
          document.getElementById("advice-text").textContent = "Please check your input and try again.";
          document.getElementById("status-text").textContent = "Error";
          document.getElementById("progress-bar-span").style.width = "0%";
        });
    });
  }

  // --- Clear button ---
  if (clearBtn) {
    clearBtn.addEventListener("click", function () {
      document.getElementById("status-text").textContent = "Awaiting result...";
      document.getElementById("progress-bar-span").style.width = "0%";
      document.getElementById("result-text").textContent = "No prediction yet";
      document.getElementById("advice-text").textContent = "";
      document.getElementById("result-box").className = "prediction-box";

      const healthIcon = document.querySelector(".health-icon");
      if (healthIcon) {
        healthIcon.innerHTML = medicalBriefcaseSVG;
        healthIcon.classList.remove("high-risk", "moderate-risk", "fair-risk", "non-diabetic");
        healthIcon.classList.add("default-icon");
      }
      
      // Clear form
      const form = document.getElementById("prediction-form");
      if (form) {
        form.reset();
      }
    });
  }

  // --- Simulate button(s) ---
  const simulateBtn = document.getElementById("simulate-btn");
  if (simulateBtn) {
    simulateBtn.addEventListener("click", () => {
      const options = [simulateNonDiabetic, simulateFairRisk, simulateModerateRisk, simulateHighRisk];
      const randomProfile = options[Math.floor(Math.random() * options.length)];
      randomProfile();
      scrollToPredict();
    });
  }

  const simulateBtnMobile = document.getElementById("simulate-btn-mobile");
  if (simulateBtnMobile) {
    simulateBtnMobile.addEventListener("click", () => {
      const options = [simulateNonDiabetic, simulateFairRisk, simulateModerateRisk, simulateHighRisk];
      const randomProfile = options[Math.floor(Math.random() * options.length)];
      randomProfile();
      scrollToPredict();

      const mobileMenu = document.getElementById("mobile-menu");
      if (mobileMenu) {
        mobileMenu.classList.remove("show");
      }
    });
  }
});

// --- Helpers ---
const getRandomInRange = (min, max, decimals = 0) => {
  const num = Math.random() * (max - min) + min;
  return decimals ? Number(num.toFixed(decimals)) : Math.floor(num);
};
const getRandomOption = (options) => options[getRandomInRange(0, options.length)];
const setFieldValue = (name, value) => {
  const element = document.querySelector(`[name="${name}"]`);
  if (element) {
    element.value = value;
    console.log(`Set ${name} to ${value}`);
  } else {
    console.warn(`Field ${name} not found`);
  }
};

// --- Updated Simulation Functions with more extreme values ---
function simulateNonDiabetic() {
  console.log('Simulating non-diabetic profile...');
  setFieldValue("Age", getRandomInRange(18, 30)); // Very young
  setFieldValue("BMI", getRandomInRange(18.5, 23, 1)); // Normal weight
  setFieldValue("Blood Glucose", getRandomInRange(70, 85)); // Excellent glucose
  setFieldValue("Blood Pressure", getRandomInRange(90, 110)); // Good BP
  setFieldValue("HbA1c", getRandomInRange(4.0, 5.2, 1)); // Excellent HbA1c
  setFieldValue("Insulin Level", getRandomInRange(2, 20)); // Low insulin
  setFieldValue("Skin thickness", getRandomInRange(10, 20)); // Thin
  setFieldValue("Pregnancies", getRandomInRange(0, 1)); // Few pregnancies
  setFieldValue("Family history", "0"); // No family history
  setFieldValue("Physical Activity", "High"); // Very active
  setFieldValue("Smoking status", "Non-Smoker"); // Non-smoker
  setFieldValue("Alcohol Intake", getRandomInRange(0, 2)); // Very low alcohol
  setFieldValue("Diet_Type", getRandomOption(["Vegetarian", "Vegan"])); // Healthy diet
  setFieldValue("Cholesterol", getRandomInRange(120, 180)); // Good cholesterol
  setFieldValue("Triglycerides", getRandomInRange(50, 120)); // Low triglycerides
  setFieldValue("Waist ratio", getRandomInRange(65, 80)); // Small waist
}

function simulateFairRisk() {
  console.log('Simulating fair risk profile...');
  setFieldValue("Age", getRandomInRange(30, 40)); // Middle age
  setFieldValue("BMI", getRandomInRange(24, 27, 1)); // Slightly overweight
  setFieldValue("Blood Glucose", getRandomInRange(95, 105)); // Slightly elevated
  setFieldValue("Blood Pressure", getRandomInRange(115, 125)); // Slightly high
  setFieldValue("HbA1c", getRandomInRange(5.3, 5.8, 1)); // Normal-high
  setFieldValue("Insulin Level", getRandomInRange(30, 60)); // Moderate insulin
  setFieldValue("Skin thickness", getRandomInRange(22, 28)); // Moderate
  setFieldValue("Pregnancies", getRandomInRange(1, 3)); // Some pregnancies
  setFieldValue("Family history", getRandomOption(["0", "1"])); // Maybe family history
  setFieldValue("Physical Activity", getRandomOption(["Medium", "High"])); // Moderate activity
  setFieldValue("Smoking status", "Non-Smoker"); // Non-smoker
  setFieldValue("Alcohol Intake", getRandomInRange(3, 7)); // Moderate alcohol
  setFieldValue("Diet_Type", getRandomOption(["Vegetarian", "Non-Vegetarian"])); // Mixed diet
  setFieldValue("Cholesterol", getRandomInRange(180, 210)); // Borderline
  setFieldValue("Triglycerides", getRandomInRange(120, 170)); // Borderline
  setFieldValue("Waist ratio", getRandomInRange(80, 95)); // Moderate waist
}

function simulateModerateRisk() {
  console.log('Simulating moderate risk profile...');
  setFieldValue("Age", getRandomInRange(45, 60)); // Older
  setFieldValue("BMI", getRandomInRange(28, 32, 1)); // Overweight to obese
  setFieldValue("Blood Glucose", getRandomInRange(110, 130)); // Pre-diabetic range
  setFieldValue("Blood Pressure", getRandomInRange(130, 145)); // High
  setFieldValue("HbA1c", getRandomInRange(5.8, 6.3, 1)); // Pre-diabetic
  setFieldValue("Insulin Level", getRandomInRange(70, 120)); // Elevated
  setFieldValue("Skin thickness", getRandomInRange(30, 40)); // Thick
  setFieldValue("Pregnancies", getRandomInRange(2, 5)); // Multiple pregnancies
  setFieldValue("Family history", getRandomOption(["0", "1"])); // Maybe family history
  setFieldValue("Physical Activity", getRandomOption(["Low", "Medium"])); // Less active
  setFieldValue("Smoking status", getRandomOption(["Smoker", "Non-Smoker"])); // Maybe smoker
  setFieldValue("Alcohol Intake", getRandomInRange(7, 12)); // Higher alcohol
  setFieldValue("Diet_Type", "Non-Vegetarian"); // Less healthy diet
  setFieldValue("Cholesterol", getRandomInRange(220, 250)); // High cholesterol
  setFieldValue("Triglycerides", getRandomInRange(180, 230)); // High triglycerides
  setFieldValue("Waist ratio", getRandomInRange(95, 110)); // Large waist
}

function simulateHighRisk() {
  console.log('Simulating high risk profile...');
  setFieldValue("Age", getRandomInRange(60, 80)); // Elderly
  setFieldValue("BMI", getRandomInRange(32, 45, 1)); // Obese
  setFieldValue("Blood Glucose", getRandomInRange(140, 220)); // Diabetic range
  setFieldValue("Blood Pressure", getRandomInRange(150, 180)); // Very high BP
  setFieldValue("HbA1c", getRandomInRange(6.8, 10.0, 1)); // Diabetic range
  setFieldValue("Insulin Level", getRandomInRange(120, 250)); // Very high insulin
  setFieldValue("Skin thickness", getRandomInRange(40, 65)); // Very thick
  setFieldValue("Pregnancies", getRandomInRange(3, 8)); // Many pregnancies
  setFieldValue("Family history", "1"); // Family history present
  setFieldValue("Physical Activity", "Low"); // Sedentary
  setFieldValue("Smoking status", "Smoker"); // Smoker
  setFieldValue("Alcohol Intake", getRandomInRange(12, 20)); // High alcohol
  setFieldValue("Diet_Type", "Non-Vegetarian"); // Unhealthy diet
  setFieldValue("Cholesterol", getRandomInRange(260, 320)); // Very high cholesterol
  setFieldValue("Triglycerides", getRandomInRange(250, 400)); // Very high triglycerides
  setFieldValue("Waist ratio", getRandomInRange(110, 135)); // Very large waist
}

// --- Scroll helper ---
function scrollToPredict() {
  const predictBtn = document.querySelector('form button[type="submit"]');
  if (predictBtn) {
    predictBtn.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}