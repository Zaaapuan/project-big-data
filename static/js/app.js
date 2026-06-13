const form = document.querySelector("#prediction-form");
const predictButton = document.querySelector("#predict-button");
const errorBanner = document.querySelector("#error-banner");
const errorMessage = document.querySelector("#error-message");
const traceIntro = document.querySelector("#trace-intro");
const pipelineSteps = document.querySelector("#pipeline-steps");
const runState = document.querySelector("#run-state");
const resultSection = document.querySelector("#result-section");

const fieldMap = {
  age: "age",
  years_experience: "years-experience",
  education_level: "education-level",
  department: "department",
};

const stepNames = ["validation", "preprocessing", "kmeans", "svm"];
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const stepDelay = reducedMotion ? 120 : 700;

// General utilities ---------------------------------------------------------

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function wait(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function setModelStatus(state, text) {
  const status = document.querySelector("#model-status");
  status.classList.toggle("error", state === "error");
  document.querySelector("#status-text").textContent = text;
}

async function loadModelInfo() {
  try {
    const response = await fetch("/api/model-info");
    if (!response.ok) {
      throw new Error("Informasi model tidak tersedia.");
    }

    const info = await response.json();
    document.querySelector("#dataset-rows").textContent =
      new Intl.NumberFormat("id-ID").format(info.dataset_rows);
    document.querySelector("#category-count").textContent = info.categories.length;
    document.querySelector("#model-accuracy").textContent =
      formatPercent(info.metrics.svm_balanced_accuracy);
    document.querySelector("#pipeline-version").textContent = info.pipeline_version;
    setModelStatus("ready", "MODEL READY");
  } catch (error) {
    setModelStatus("error", "MODEL ERROR");
    showBanner(error.message);
  }
}

// Form validation ----------------------------------------------------------

function clearErrors() {
  errorBanner.hidden = true;
  Object.values(fieldMap).forEach((fieldId) => {
    const field = document.querySelector(`#${fieldId}`);
    field.closest(".field-group").classList.remove("has-error");
    field.removeAttribute("aria-invalid");
    document.querySelector(`#${fieldId}-error`).textContent = "";
  });
}

function showBanner(message) {
  errorMessage.textContent = message;
  errorBanner.hidden = false;
}

function showFieldErrors(errors) {
  Object.entries(errors).forEach(([apiField, message]) => {
    const fieldId = fieldMap[apiField];
    if (!fieldId) {
      return;
    }

    const field = document.querySelector(`#${fieldId}`);
    field.closest(".field-group").classList.add("has-error");
    field.setAttribute("aria-invalid", "true");
    document.querySelector(`#${fieldId}-error`).textContent = message;
  });
}

function readPayload() {
  return {
    age: Number(document.querySelector("#age").value),
    years_experience: Number(document.querySelector("#years-experience").value),
    education_level: Number(document.querySelector("#education-level").value),
    department: document.querySelector("#department").value,
  };
}

function setLoading(isLoading) {
  predictButton.disabled = isLoading;
  predictButton.classList.toggle("loading", isLoading);
  predictButton.querySelector(".button-label").textContent =
    isLoading ? "MEMPROSES DATA" : "PROSES DATA";
}

// Educational execution trace ---------------------------------------------

function getStep(stepName) {
  return document.querySelector(`[data-step="${stepName}"]`);
}

function resetPipeline() {
  stepNames.forEach((stepName) => {
    const step = getStep(stepName);
    step.classList.remove("active", "complete");
    step.querySelector(".step-status").textContent = "MENUNGGU";
    step.querySelector(".step-output").innerHTML = "";
  });

  traceIntro.hidden = false;
  pipelineSteps.hidden = true;
  resultSection.hidden = true;
  runState.textContent = "IDLE";
  runState.className = "run-state";
}

function startPipeline() {
  resetPipeline();
  traceIntro.hidden = true;
  pipelineSteps.hidden = false;
  runState.textContent = "RUNNING";
  runState.className = "run-state running";
}

function activateStep(stepName) {
  const step = getStep(stepName);
  step.classList.add("active");
  step.querySelector(".step-status").textContent = "BERJALAN";
}

function completeStep(stepName, outputHtml) {
  const step = getStep(stepName);
  step.querySelector(".step-output").innerHTML = outputHtml;
  step.classList.remove("active");
  step.classList.add("complete");
  step.querySelector(".step-status").textContent = "SELESAI";
}

async function revealStep(stepName, outputHtml) {
  activateStep(stepName);
  await wait(stepDelay);
  completeStep(stepName, outputHtml);
}

function renderValidationOutput(result) {
  const summary = result.input_summary;
  return `
    <p class="output-message">${escapeHtml(result.process.input_validation.message)}</p>
    <pre class="code-output">age=${summary.age}
years_experience=${summary.years_experience}
education_level=${summary.education_level}
department="${escapeHtml(summary.department)}"</pre>
  `;
}

function renderPreprocessingOutput(preprocessing) {
  const formulas = preprocessing.numeric_scaling
    .map((item) => `${item.feature}: ${item.formula}`)
    .join("\n");

  const encoding = Object.entries(preprocessing.department_encoding.one_hot)
    .map(([department, value]) => `${department}=${value}`)
    .join(", ");

  return `
    <p class="output-message">
      Fitur numerik diubah ke skala yang sebanding; departemen diubah menjadi
      angka biner.
    </p>
    <pre class="code-output">${escapeHtml(formulas)}

one_hot: ${escapeHtml(encoding)}
vector: [${preprocessing.transformed_vector.join(", ")}]</pre>
  `;
}

function renderKmeansOutput(kmeans) {
  const maximumDistance = Math.max(
    ...kmeans.distances.map((item) => item.distance),
  );
  const rows = kmeans.distances.map((item) => {
    const closeness = maximumDistance === 0
      ? 100
      : Math.max(8, 100 - (item.distance / maximumDistance) * 88);
    const selectedClass =
      item.cluster_id === kmeans.nearest_cluster_id ? "selected" : "";

    return `
      <div class="distance-row ${selectedClass}">
        <span>${escapeHtml(item.category)}</span>
        <span class="mini-track"><span style="width:${closeness}%"></span></span>
        <code>${item.distance.toFixed(4)}</code>
      </div>
    `;
  }).join("");

  return `
    <p class="output-message">
      Jarak paling kecil dipilih: <strong>${escapeHtml(kmeans.nearest_category)}</strong>.
    </p>
    <div class="distance-list">${rows}</div>
  `;
}

function renderSvmOutput(svm) {
  const rows = svm.probabilities.map((item) => {
    const selectedClass =
      item.cluster_id === svm.selected_cluster_id ? "selected" : "";
    const percentage = Math.round(item.probability * 100);

    return `
      <div class="probability-row ${selectedClass}">
        <span>${escapeHtml(item.category)}</span>
        <span class="mini-track"><span style="width:${percentage}%"></span></span>
        <code>${percentage}%</code>
      </div>
    `;
  }).join("");

  return `
    <p class="output-message">
      Probabilitas tertinggi menjadi hasil SVM:
      <strong>${escapeHtml(svm.selected_category)}</strong>.
    </p>
    <div class="probability-list">${rows}</div>
  `;
}

function summaryItem(label, value) {
  return `
    <div class="summary-item">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
    </div>
  `;
}

function renderFinalResult(result) {
  document.querySelector("#result-category").textContent = result.category;
  document.querySelector("#result-description").textContent = result.description;
  document.querySelector("#svm-result").textContent = result.svm.category;
  document.querySelector("#kmeans-result").textContent = result.kmeans.category;
  document.querySelector("#result-disclaimer").textContent = result.disclaimer;

  const agreement = document.querySelector("#agreement-badge");
  agreement.textContent = result.models_agree
    ? "MODELS AGREE"
    : "MODELS DIFFER";
  agreement.classList.toggle("disagree", !result.models_agree);

  const confidence = Math.round(result.svm.confidence * 100);
  document.querySelector("#confidence-value").textContent = `${confidence}%`;
  const confidenceTrack = document.querySelector("#confidence-track");
  confidenceTrack.setAttribute("aria-valuenow", String(confidence));
  const confidenceBar = document.querySelector("#confidence-bar");
  confidenceBar.style.width = "0%";

  const summary = result.input_summary;
  document.querySelector("#input-summary").innerHTML = [
    summaryItem("Umur", `${summary.age} tahun`),
    summaryItem("Pengalaman", `${summary.years_experience} tahun`),
    summaryItem("Pendidikan", summary.education_label),
    summaryItem("Departemen", summary.department),
  ].join("");

  resultSection.hidden = false;
  requestAnimationFrame(() => {
    confidenceBar.style.width = `${confidence}%`;
  });
  resultSection.scrollIntoView({behavior: reducedMotion ? "auto" : "smooth"});
}

async function playEducationalTrace(result) {
  completeStep("validation", renderValidationOutput(result));
  await wait(stepDelay);

  await revealStep(
    "preprocessing",
    renderPreprocessingOutput(result.process.preprocessing),
  );
  await revealStep(
    "kmeans",
    renderKmeansOutput(result.process.kmeans),
  );
  await revealStep(
    "svm",
    renderSvmOutput(result.process.svm),
  );

  runState.textContent = "COMPLETE";
  runState.className = "run-state complete";
  renderFinalResult(result);
}

// Main interaction ---------------------------------------------------------

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearErrors();

  if (!form.checkValidity()) {
    form.reportValidity();
    showBanner("Lengkapi seluruh kolom dengan nilai yang valid.");
    return;
  }

  setLoading(true);
  startPipeline();
  activateStep("validation");

  try {
    const predictionRequest = fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(readPayload()),
    });
    await wait(stepDelay);
    const response = await predictionRequest;
    const result = await response.json();

    if (!response.ok) {
      showBanner(result.message || "Prediksi tidak dapat diproses.");
      if (result.fields) {
        showFieldErrors(result.fields);
      }
      resetPipeline();
      return;
    }

    await playEducationalTrace(result);
  } catch (_error) {
    showBanner("Tidak dapat terhubung ke model lokal. Jalankan ulang aplikasi.");
    resetPipeline();
  } finally {
    setLoading(false);
  }
});

Object.values(fieldMap).forEach((fieldId) => {
  document.querySelector(`#${fieldId}`).addEventListener("input", () => {
    const field = document.querySelector(`#${fieldId}`);
    field.closest(".field-group").classList.remove("has-error");
    field.removeAttribute("aria-invalid");
    document.querySelector(`#${fieldId}-error`).textContent = "";
  });
});

resetPipeline();
loadModelInfo();
