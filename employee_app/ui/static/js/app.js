const form = document.querySelector("#prediction-form");
const predictButton = document.querySelector("#predict-button");
const errorBanner = document.querySelector("#error-banner");
const errorMessage = document.querySelector("#error-message");
const screens = [...document.querySelectorAll(".wizard-screen")];
const stepperItems = [...document.querySelectorAll(".stepper-item")];
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const fieldMap = {
  age: "age",
  years_experience: "years-experience",
  education_level: "education-level",
  department: "department",
};

let currentStep = 0;
let predictionResult = null;

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
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
    document.querySelector("#pipeline-version").textContent = info.pipeline_version;
    setModelStatus("ready", "MODEL READY");
  } catch (error) {
    setModelStatus("error", "MODEL ERROR");
    showBanner(error.message);
  }
}

function showStep(stepIndex) {
  if (stepIndex > 0 && !predictionResult) {
    return;
  }

  currentStep = stepIndex;
  screens.forEach((screen, index) => {
    const isActive = index === stepIndex;
    screen.hidden = !isActive;
    screen.classList.toggle("active", isActive);
  });

  stepperItems.forEach((item, index) => {
    const isActive = index === stepIndex;
    item.classList.toggle("active", isActive);
    item.classList.toggle("complete", Boolean(predictionResult) && index < stepIndex);
    item.disabled = !predictionResult && index > 0;
    if (isActive) {
      item.setAttribute("aria-current", "step");
    } else {
      item.removeAttribute("aria-current");
    }
  });

  document.querySelector(".wizard-card").scrollIntoView({
    behavior: reducedMotion ? "auto" : "smooth",
    block: "start",
  });
}

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
    isLoading ? "MEMPROSES DATA" : "MULAI ANALISIS";
}

function renderPreprocessing(preprocessing) {
  document.querySelector("#scaling-output").innerHTML =
    preprocessing.numeric_scaling.map((item) => `
      <div>
        <span>${escapeHtml(item.feature)}</span>
        <code>${escapeHtml(item.formula)}</code>
      </div>
    `).join("");

  document.querySelector("#encoding-output").innerHTML =
    Object.entries(preprocessing.department_encoding.one_hot)
      .map(([department, value]) => `
        <div class="${value === 1 ? "active" : ""}">
          <span>${escapeHtml(department)}</span>
          <strong>${value}</strong>
        </div>
      `).join("");

  document.querySelector("#transformed-vector").textContent =
    `[${preprocessing.transformed_vector.join(", ")}]`;
}

function renderModelBars(items, valueKey, selectedClusterId) {
  const maximum = Math.max(...items.map((item) => item[valueKey]));
  return items.map((item) => {
    const selected = item.cluster_id === selectedClusterId;
    const displayValue = valueKey === "probability"
      ? formatPercent(item[valueKey])
      : item[valueKey].toFixed(4);
    const width = valueKey === "probability"
      ? item[valueKey] * 100
      : Math.max(8, (item[valueKey] / maximum) * 100);

    return `
      <div class="model-bar ${selected ? "selected" : ""}">
        <div>
          <span>${escapeHtml(item.category)}</span>
          <code>${displayValue}</code>
        </div>
        <div class="bar-track"><span style="width:${width}%"></span></div>
      </div>
    `;
  }).join("");
}

function summaryItem(label, value) {
  return `
    <div>
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
    </div>
  `;
}

function renderResult(result) {
  renderPreprocessing(result.process.preprocessing);

  const kmeans = result.process.kmeans;
  document.querySelector("#kmeans-output").innerHTML = renderModelBars(
    kmeans.distances,
    "distance",
    kmeans.nearest_cluster_id,
  );
  document.querySelector("#kmeans-selection").innerHTML =
    `Jarak terkecil memilih cluster <strong>${escapeHtml(kmeans.nearest_category)}</strong>.`;

  const svm = result.process.svm;
  document.querySelector("#svm-output").innerHTML = renderModelBars(
    svm.probabilities,
    "probability",
    svm.selected_cluster_id,
  );
  document.querySelector("#svm-selection").innerHTML =
    `Confidence tertinggi memilih <strong>${escapeHtml(svm.selected_category)}</strong>.`;

  document.querySelector("#result-category").textContent = result.category;
  document.querySelector("#result-description").textContent = result.description;
  document.querySelector("#svm-result").textContent = result.svm.category;
  document.querySelector("#kmeans-result").textContent = result.kmeans.category;
  document.querySelector("#result-disclaimer").textContent = result.disclaimer;

  const agreement = document.querySelector("#agreement-badge");
  agreement.textContent = result.models_agree ? "MODEL SEPAKAT" : "MODEL BERBEDA";
  agreement.classList.toggle("disagree", !result.models_agree);

  const confidence = Math.round(result.svm.confidence * 100);
  document.querySelector("#confidence-value").textContent = `${confidence}%`;
  document.querySelector("#confidence-track").setAttribute(
    "aria-valuenow",
    String(confidence),
  );
  document.querySelector("#confidence-bar").style.width = `${confidence}%`;

  const summary = result.input_summary;
  document.querySelector("#input-summary").innerHTML = [
    summaryItem("Umur", `${summary.age} tahun`),
    summaryItem("Pengalaman", `${summary.years_experience} tahun`),
    summaryItem("Pendidikan", summary.education_label),
    summaryItem("Departemen", summary.department),
  ].join("");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearErrors();

  if (!form.checkValidity()) {
    form.reportValidity();
    showBanner("Lengkapi seluruh kolom dengan nilai yang valid.");
    return;
  }

  setLoading(true);
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(readPayload()),
    });
    const result = await response.json();

    if (!response.ok) {
      showBanner(result.message || "Prediksi tidak dapat diproses.");
      if (result.fields) {
        showFieldErrors(result.fields);
      }
      return;
    }

    predictionResult = result;
    renderResult(result);
    showStep(1);
  } catch (_error) {
    showBanner("Tidak dapat terhubung ke model lokal. Jalankan ulang aplikasi.");
  } finally {
    setLoading(false);
  }
});

document.querySelectorAll("[data-go-step]").forEach((button) => {
  button.addEventListener("click", () => {
    showStep(Number(button.dataset.goStep));
  });
});

stepperItems.forEach((item) => {
  item.addEventListener("click", () => {
    if (predictionResult) {
      showStep(Number(item.dataset.stepTarget));
    }
  });
});

document.querySelector("#restart-button").addEventListener("click", () => {
  predictionResult = null;
  document.querySelector("#confidence-bar").style.width = "0%";
  clearErrors();
  showStep(0);
  document.querySelector("#age").focus();
});

Object.values(fieldMap).forEach((fieldId) => {
  document.querySelector(`#${fieldId}`).addEventListener("input", () => {
    const field = document.querySelector(`#${fieldId}`);
    field.closest(".field-group").classList.remove("has-error");
    field.removeAttribute("aria-invalid");
    document.querySelector(`#${fieldId}-error`).textContent = "";
  });
});

showStep(0);
loadModelInfo();
