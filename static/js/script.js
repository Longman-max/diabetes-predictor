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

  healthIcon.classList.remove(
    "high-risk",
    "moderate-risk",
    "fair-risk",
    "non-diabetic",
    "default-icon"
  );
  healthIcon.classList.add(className);

  if (
    className === "high-risk" ||
    className === "moderate-risk" ||
    className === "fair-risk"
  ) {
    healthIcon.innerHTML = alertTriangleSVG;
  } else if (className === "non-diabetic") {
    healthIcon.innerHTML = checkCircleSVG;
  } else {
    healthIcon.innerHTML = medicalBriefcaseSVG;
  }
}

// --- Function to display results ---
function displayPredictionResult(prediction, advice, className) {
  console.log("Displaying result:", { prediction, advice, className });

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
    "Age",
    "Weight",
    "Blood Glucose",
    "HbA1c",
    "race",
    "gender",
  ];

  requiredFields.forEach((field) => {
    const value = formData.get(field);
    if (!value || value.trim() === "") {
      errors.push(`${field} is required`);
    }
  });

  // Numeric range validations
  const age = parseFloat(formData.get("Age"));
  if (age < 0 || age > 100) {
    errors.push("Age must be between 0-100");
  }

  const weight = parseFloat(formData.get("Weight"));
  if (weight < 20 || weight > 200) {
    errors.push("Weight must be between 20-200 kg");
  }

  const glucose = parseFloat(formData.get("Blood Glucose"));
  if (glucose < 50 || glucose > 400) {
    errors.push("Blood Glucose must be between 50-400 mg/dL");
  }

  const hba1c = parseFloat(formData.get("HbA1c"));
  if (hba1c < 4.0 || hba1c > 14.0) {
    errors.push("HbA1c must be between 4.0-14.0%");
  }

  // Optional numeric validations
  const outpatient = parseFloat(formData.get("number_outpatient"));
  if (!isNaN(outpatient) && (outpatient < 0 || outpatient > 50)) {
    errors.push("Number of Outpatient Visits must be between 0-50");
  }

  const emergency = parseFloat(formData.get("number_emergency"));
  if (!isNaN(emergency) && (emergency < 0 || emergency > 20)) {
    errors.push("Number of Emergency Visits must be between 0-20");
  }

  const inpatient = parseFloat(formData.get("number_inpatient"));
  if (!isNaN(inpatient) && (inpatient < 0 || inpatient > 20)) {
    errors.push("Number of Inpatient Visits must be between 0-20");
  }

  const medications = parseFloat(formData.get("num_medications"));
  if (!isNaN(medications) && (medications < 0 || medications > 50)) {
    errors.push("Number of Medications must be between 0-50");
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
        alert(
          "Please fix the following errors:\n" + validationErrors.join("\n")
        );
        return;
      }

      console.log(
        "Submitting form data:",
        Object.fromEntries(formData.entries())
      );

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
          console.log("Response status:", response.status);
          if (!response.ok) {
            return response.json().then((err) => Promise.reject(err));
          }
          return response.json();
        })
        .then((data) => {
          console.log("Received data:", data);
          if (data.error) {
            throw new Error(data.error);
          }
          displayPredictionResult(data.message, data.advice, data.class_name);
        })
        .catch((error) => {
          console.error("Error:", error);
          document.getElementById("result-text").textContent =
            "Prediction failed: " +
            (error.message || error.error || "Unknown error");
          document.getElementById("advice-text").textContent =
            "Please check your input and try again.";
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
        healthIcon.classList.remove(
          "high-risk",
          "moderate-risk",
          "fair-risk",
          "non-diabetic"
        );
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
      const options = [
        simulateNonDiabetic,
        simulateFairRisk,
        simulateModerateRisk,
        simulateHighRisk,
      ];
      const randomProfile = options[Math.floor(Math.random() * options.length)];
      randomProfile();
      scrollToPredict();
    });
  }

  const simulateBtnMobile = document.getElementById("simulate-btn-mobile");
  if (simulateBtnMobile) {
    simulateBtnMobile.addEventListener("click", () => {
      const options = [
        simulateNonDiabetic,
        simulateFairRisk,
        simulateModerateRisk,
        simulateHighRisk,
      ];
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
const getRandomOption = (options) =>
  options[getRandomInRange(0, options.length)];
const setFieldValue = (name, value) => {
  const element = document.querySelector(`[name="${name}"]`);
  if (element) {
    element.value = value;
    console.log(`Set ${name} to ${value}`);
  } else {
    console.warn(`Field ${name} not found`);
  }
};

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
    Age: () => getRandomInRange(20, 80),
    Weight: () => getRandomInRange(50, 120),
    "Blood Glucose": () => getRandomInRange(70, 300),
    HbA1c: () => getRandomInRange(5.0, 10.0, 1),
    number_outpatient: () => getRandomInRange(0, 10),
    number_emergency: () => getRandomInRange(0, 5),
    number_inpatient: () => getRandomInRange(0, 3),
    num_medications: () => getRandomInRange(0, 10),
    race: () =>
      getRandomOption([
        "Caucasian",
        "AfricanAmerican",
        "Asian",
        "Hispanic",
        "Other",
      ]),
    gender: () => getRandomOption(["Female", "Male", "Other"]),
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
  scrollToPredict();
}

// --- Scroll helper ---
function scrollToPredict() {
  const predictBtn = document.querySelector('form button[type="submit"]');
  if (predictBtn) {
    predictBtn.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}
