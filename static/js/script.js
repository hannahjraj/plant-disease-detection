/**
 * Plant Disease Detection - Interactive Client Logic
 * Handles file drag-and-drop, asynchronous model inference,
 * Chart.js visualizations, and dynamic UI states.
 */

document.addEventListener("DOMContentLoaded", () => {
  initMobileNav();
  initPredictionInterface();
  initPerformanceCharts();
  initDiseaseSearch();
});

/* -------------------------------------------------------------
 * 1. Mobile Navigation
 * ------------------------------------------------------------- */
function initMobileNav() {
  const toggleBtn = document.getElementById("navToggle");
  const navMenu = document.getElementById("navMenu");

  if (toggleBtn && navMenu) {
    toggleBtn.addEventListener("click", () => {
      navMenu.classList.toggle("open");
    });

    // Close menu when clicking outside
    document.addEventListener("click", (e) => {
      if (!toggleBtn.contains(e.target) && !navMenu.contains(e.target)) {
        navMenu.classList.remove("open");
      }
    });
  }
}

/* -------------------------------------------------------------
 * 2. Prediction Interface (Upload, Drag-and-drop, AJAX Detect)
 * ------------------------------------------------------------- */
function initPredictionInterface() {
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const previewContainer = document.getElementById("previewContainer");
  const previewImg = document.getElementById("previewImg");
  const previewName = document.getElementById("previewName");
  const previewSize = document.getElementById("previewSize");
  const btnRemove = document.getElementById("btnRemove");
  const btnDetect = document.getElementById("btnDetect");
  const spinnerContainer = document.getElementById("spinnerContainer");
  const resultCard = document.getElementById("resultCard");
  const alertContainer = document.getElementById("alertContainer");

  let currentFile = null;
  let currentSampleImage = null;

  if (!dropZone || !fileInput) return;

  // Open file dialog when clicking drop zone
  dropZone.addEventListener("click", () => {
    fileInput.click();
  });

  // Drag & Drop events
  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files.length > 0) {
      handleFileSelected(dt.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Sample Leaf buttons
  document.querySelectorAll(".sample-pill-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const sampleFile = btn.getAttribute("data-sample");
      const sampleTitle = btn.getAttribute("data-title");
      if (sampleFile) {
        currentSampleImage = sampleFile;
        currentFile = null;
        fileInput.value = "";

        // Preview sample image
        const imgUrl = `/static/images/${sampleFile}`;
        previewImg.src = imgUrl;
        previewName.textContent = sampleTitle || sampleFile;
        previewSize.textContent = "Preset Sample Leaf (Preloaded)";
        previewContainer.style.display = "block";
        dropZone.style.display = "none";
        hideAlert();
        if (resultCard) resultCard.style.display = "none";
      }
    });
  });

  // Handle selected image file
  function handleFileSelected(file) {
    // Validate format
    const validTypes = ["image/jpeg", "image/png", "image/webp", "image/jpg"];
    if (!validTypes.includes(file.type.toLowerCase())) {
      showAlert("Unsupported file format. Please upload a JPG, JPEG, PNG, or WEBP image.", "error");
      return;
    }

    // Validate size (16MB)
    if (file.size > 16 * 1024 * 1024) {
      showAlert("File size exceeds 16MB limit. Please upload a smaller image.", "error");
      return;
    }

    currentFile = file;
    currentSampleImage = null;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewName.textContent = file.name;
      previewSize.textContent = formatBytes(file.size);
      previewContainer.style.display = "block";
      dropZone.style.display = "none";
      hideAlert();
      if (resultCard) resultCard.style.display = "none";
    };
    reader.readAsDataURL(file);
  }

  // Remove preview
  if (btnRemove) {
    btnRemove.addEventListener("click", () => {
      currentFile = null;
      currentSampleImage = null;
      fileInput.value = "";
      previewImg.src = "";
      previewContainer.style.display = "none";
      dropZone.style.display = "block";
      if (resultCard) resultCard.style.display = "none";
      hideAlert();
    });
  }

  // Detect button trigger
  if (btnDetect) {
    btnDetect.addEventListener("click", async () => {
      if (!currentFile && !currentSampleImage) {
        showAlert("Please select or upload a plant leaf image first.", "warning");
        return;
      }

      const formData = new FormData();
      if (currentFile) {
        formData.append("file", currentFile);
      } else if (currentSampleImage) {
        formData.append("sample_image", currentSampleImage);
      }

      // UI state during detection
      hideAlert();
      if (resultCard) resultCard.style.display = "none";
      spinnerContainer.style.display = "block";
      btnDetect.disabled = true;
      btnDetect.style.opacity = "0.7";

      try {
        const response = await fetch("/predict", {
          method: "POST",
          body: formData
        });

        const data = await response.json();
        spinnerContainer.style.display = "none";
        btnDetect.disabled = false;
        btnDetect.style.opacity = "1";

        if (!data.success) {
          if (data.model_ready === false) {
            showAlert(
              `<strong>Model Not Trained:</strong> ${data.message} <br><br><code>python train_model.py</code>`,
              "warning"
            );
          } else {
            showAlert(data.message || "An error occurred during prediction.", "error");
          }
          return;
        }

        // Render prediction result
        displayPredictionResult(data);

      } catch (err) {
        console.error("Prediction request failed:", err);
        spinnerContainer.style.display = "none";
        btnDetect.disabled = false;
        btnDetect.style.opacity = "1";
        showAlert("Server connection failed. Make sure Flask is running at http://127.0.0.1:5000.", "error");
      }
    });
  }

  function displayPredictionResult(res) {
    if (!resultCard) return;

    // Populate data
    document.getElementById("resImage").src = res.image_url;
    document.getElementById("resDiseaseName").textContent = res.disease_name;
    document.getElementById("resPlant").textContent = `Plant: ${res.plant}`;

    // Status Badge
    const statusBadge = document.getElementById("resStatusBadge");
    statusBadge.textContent = res.status;
    statusBadge.className = `badge badge-${res.badge_color || "green"}`;

    // Confidence
    const confScore = res.confidence;
    document.getElementById("resConfidenceVal").textContent = `${confScore}%`;
    const confFill = document.getElementById("resConfidenceFill");
    confFill.style.width = "0%";
    setTimeout(() => {
      confFill.style.width = `${Math.min(100, Math.max(5, confScore))}%`;
    }, 100);

    // Description
    document.getElementById("resDescription").textContent = res.description;

    // Symptoms List
    const symptomsEl = document.getElementById("resSymptomsList");
    symptomsEl.innerHTML = "";
    (res.symptoms || []).forEach((sym) => {
      const li = document.createElement("li");
      li.textContent = sym;
      symptomsEl.appendChild(li);
    });

    // Recommendations List
    const recsEl = document.getElementById("resRecsList");
    recsEl.innerHTML = "";
    (res.recommendations || []).forEach((rec) => {
      const li = document.createElement("li");
      li.textContent = rec;
      recsEl.appendChild(li);
    });

    // Top Predictions Breakdown if available
    const topContainer = document.getElementById("resTopPredictions");
    if (topContainer && res.top_predictions && res.top_predictions.length > 1) {
      topContainer.innerHTML = "";
      res.top_predictions.forEach((item) => {
        const row = document.createElement("div");
        row.className = "top-pred-row";
        row.innerHTML = `
          <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:4px;">
            <span>${item.class_name}</span>
            <span style="font-weight:700;">${item.confidence}%</span>
          </div>
          <div style="height:6px; background:#e2e8f0; border-radius:10px; overflow:hidden; margin-bottom:10px;">
            <div style="height:100%; width:${item.confidence}%; background:#16a34a; border-radius:10px;"></div>
          </div>
        `;
        topContainer.appendChild(row);
      });
      document.getElementById("topPredSection").style.display = "block";
    }

    resultCard.style.display = "block";
    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function showAlert(msg, type = "info") {
    if (!alertContainer) return;
    alertContainer.innerHTML = `
      <div class="alert-box alert-${type}">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="flex-shrink:0;">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="8" x2="12" y2="12"></line>
          <line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
        <div>${msg}</div>
      </div>
    `;
    alertContainer.style.display = "block";
    alertContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function hideAlert() {
    if (!alertContainer) return;
    alertContainer.style.display = "none";
    alertContainer.innerHTML = "";
  }

  function formatBytes(bytes, decimals = 1) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
  }
}

/* -------------------------------------------------------------
 * 3. Model Performance Charts (Chart.js)
 * ------------------------------------------------------------- */
function initPerformanceCharts() {
  const historyDataScript = document.getElementById("trainingHistoryData");
  if (!historyDataScript) return;

  let history;
  try {
    history = JSON.parse(historyDataScript.textContent);
  } catch (e) {
    console.warn("No training history data found on page.");
    return;
  }

  if (!history || !history.epochs || history.epochs.length === 0) return;

  const labels = history.epochs.map((ep) => `Epoch ${ep}`);

  // Accuracy Chart
  const accCtx = document.getElementById("accuracyChart");
  if (accCtx && typeof Chart !== "undefined") {
    new Chart(accCtx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Training Accuracy",
            data: history.accuracy.map((v) => (v * 100).toFixed(1)),
            borderColor: "#16a34a",
            backgroundColor: "rgba(22, 163, 74, 0.1)",
            borderWidth: 2.5,
            tension: 0.3,
            fill: true
          },
          {
            label: "Validation Accuracy",
            data: history.val_accuracy.map((v) => (v * 100).toFixed(1)),
            borderColor: "#0284c7",
            backgroundColor: "rgba(2, 132, 199, 0.08)",
            borderWidth: 2.5,
            borderDash: [5, 5],
            tension: 0.3,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${ctx.raw}%`
            }
          }
        },
        scales: {
          y: {
            title: { display: true, text: "Accuracy (%)" },
            min: 0,
            max: 100
          },
          x: {
            title: { display: true, text: "Training Epochs" }
          }
        }
      }
    });
  }

  // Loss Chart
  const lossCtx = document.getElementById("lossChart");
  if (lossCtx && typeof Chart !== "undefined") {
    new Chart(lossCtx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Training Loss",
            data: history.loss,
            borderColor: "#ea580c",
            backgroundColor: "rgba(234, 88, 12, 0.1)",
            borderWidth: 2.5,
            tension: 0.3,
            fill: true
          },
          {
            label: "Validation Loss",
            data: history.val_loss,
            borderColor: "#dc2626",
            backgroundColor: "rgba(220, 38, 38, 0.08)",
            borderWidth: 2.5,
            borderDash: [5, 5],
            tension: 0.3,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" }
        },
        scales: {
          y: {
            title: { display: true, text: "Loss (Cross-Entropy)" },
            beginAtZero: true
          },
          x: {
            title: { display: true, text: "Training Epochs" }
          }
        }
      }
    });
  }
}

/* -------------------------------------------------------------
 * 4. Disease Search & Catalog Filter
 * ------------------------------------------------------------- */
function initDiseaseSearch() {
  const searchInput = document.getElementById("diseaseSearchInput");
  const cards = document.querySelectorAll(".disease-card");

  if (!searchInput || cards.length === 0) return;

  searchInput.addEventListener("input", (e) => {
    const term = e.target.value.toLowerCase().trim();

    cards.forEach((card) => {
      const title = (card.querySelector(".disease-card-title")?.textContent || "").toLowerCase();
      const crop = (card.querySelector(".disease-crop-name")?.textContent || "").toLowerCase();
      const desc = (card.querySelector(".disease-card-desc")?.textContent || "").toLowerCase();

      if (title.includes(term) || crop.includes(term) || desc.includes(term)) {
        card.style.display = "flex";
      } else {
        card.style.display = "none";
      }
    });
  });
}
