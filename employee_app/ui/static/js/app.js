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
let modelInfo = null;

const clusterColors = {
  "Emerging Talent": "#6a5fc1",
  "Academic Achiever": "#fa7faa",
  "Seasoned Veteran": "#1f1633",
};

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
    modelInfo = info;
    document.querySelector("#pipeline-version").textContent = info.pipeline_version;
    renderBaselinePlot(info.cluster_plot);
    stepperItems[1].disabled = false;
    setModelStatus("ready", "MODEL READY");
  } catch (error) {
    setModelStatus("error", "MODEL ERROR");
    showBanner(error.message);
  }
}

function showStep(stepIndex) {
  if (stepIndex > 1 && !predictionResult) {
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
    item.classList.toggle("complete", index < stepIndex);
    item.disabled = !predictionResult && index > 1;
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

function plotBounds(plotData) {
  const coordinates = [
    ...plotData.points,
    ...plotData.centroids,
    ...(plotData.new_point ? [plotData.new_point] : []),
  ];
  const xValues = coordinates.map((point) => point.x);
  const yValues = coordinates.map((point) => point.y);
  const xPadding = (Math.max(...xValues) - Math.min(...xValues)) * 0.08 || 1;
  const yPadding = (Math.max(...yValues) - Math.min(...yValues)) * 0.1 || 1;
  return {
    minX: Math.min(...xValues) - xPadding,
    maxX: Math.max(...xValues) + xPadding,
    minY: Math.min(...yValues) - yPadding,
    maxY: Math.max(...yValues) + yPadding,
  };
}

function drawCross(context, x, y, size) {
  context.save();
  context.strokeStyle = "#ffffff";
  context.lineWidth = 8;
  context.beginPath();
  context.moveTo(x - size, y - size);
  context.lineTo(x + size, y + size);
  context.moveTo(x + size, y - size);
  context.lineTo(x - size, y + size);
  context.stroke();
  context.strokeStyle = "#150f23";
  context.lineWidth = 4;
  context.stroke();
  context.restore();
}

function drawNewPoint(context, x, y) {
  context.save();
  context.fillStyle = "#c2ef4e";
  context.strokeStyle = "#150f23";
  context.lineWidth = 4;
  context.beginPath();
  context.arc(x, y, 11, 0, Math.PI * 2);
  context.fill();
  context.stroke();
  context.beginPath();
  context.moveTo(x - 16, y);
  context.lineTo(x + 16, y);
  context.moveTo(x, y - 16);
  context.lineTo(x, y + 16);
  context.stroke();
  context.restore();
}

function drawClusterPlot(canvas, plotData, title) {
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const margin = {top: 72, right: 210, bottom: 62, left: 72};
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const bounds = plotBounds(plotData);
  const xScale = (value) => margin.left
    + ((value - bounds.minX) / (bounds.maxX - bounds.minX)) * plotWidth;
  const yScale = (value) => margin.top + plotHeight
    - ((value - bounds.minY) / (bounds.maxY - bounds.minY)) * plotHeight;

  context.clearRect(0, 0, width, height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);
  context.fillStyle = "#1f1633";
  context.font = "600 23px Rubik, Arial, sans-serif";
  context.textAlign = "center";
  context.fillText(title, width / 2, 31);
  context.fillStyle = "#686174";
  context.font = "13px Rubik, Arial, sans-serif";
  context.fillText(
    "PCA 2D dari Age, TotalWorkingYears, Education, dan Department",
    width / 2,
    52,
  );

  context.strokeStyle = "#e5e2e9";
  context.lineWidth = 1;
  context.font = "11px Monaco, monospace";
  for (let tick = 0; tick <= 5; tick += 1) {
    const x = margin.left + (plotWidth / 5) * tick;
    const y = margin.top + (plotHeight / 5) * tick;
    context.beginPath();
    context.moveTo(x, margin.top);
    context.lineTo(x, margin.top + plotHeight);
    context.moveTo(margin.left, y);
    context.lineTo(margin.left + plotWidth, y);
    context.stroke();

    const xValue = bounds.minX
      + ((bounds.maxX - bounds.minX) / 5) * tick;
    const yValue = bounds.maxY
      - ((bounds.maxY - bounds.minY) / 5) * tick;
    context.fillStyle = "#8d8796";
    context.textAlign = "center";
    context.fillText(xValue.toFixed(1), x, margin.top + plotHeight + 20);
    context.textAlign = "right";
    context.fillText(yValue.toFixed(1), margin.left - 10, y + 4);
  }

  context.save();
  context.globalAlpha = 0.62;
  plotData.points.forEach((point) => {
    const centroid = plotData.centroids.find(
      (item) => item.cluster_id === point.cluster_id,
    );
    context.fillStyle = clusterColors[centroid.category];
    context.beginPath();
    context.arc(xScale(point.x), yScale(point.y), 3.2, 0, Math.PI * 2);
    context.fill();
  });
  context.restore();

  plotData.centroids.forEach((centroid) => {
    drawCross(context, xScale(centroid.x), yScale(centroid.y), 9);
  });
  if (plotData.new_point) {
    drawNewPoint(
      context,
      xScale(plotData.new_point.x),
      yScale(plotData.new_point.y),
    );
  }

  context.fillStyle = "#686174";
  context.font = "13px Rubik, Arial, sans-serif";
  context.textAlign = "center";
  context.fillText(plotData.axis_labels[0], margin.left + plotWidth / 2, height - 13);
  context.save();
  context.translate(17, margin.top + plotHeight / 2);
  context.rotate(-Math.PI / 2);
  context.fillText(plotData.axis_labels[1], 0, 0);
  context.restore();

  const legendX = margin.left + plotWidth + 25;
  let legendY = margin.top + 10;
  context.textAlign = "left";
  context.font = "12px Rubik, Arial, sans-serif";
  plotData.centroids.forEach((centroid) => {
    context.fillStyle = clusterColors[centroid.category];
    context.beginPath();
    context.arc(legendX + 6, legendY, 5, 0, Math.PI * 2);
    context.fill();
    context.fillStyle = "#1f1633";
    context.fillText(centroid.category, legendX + 18, legendY + 4);
    legendY += 25;
  });
  drawCross(context, legendX + 6, legendY, 6);
  context.fillStyle = "#1f1633";
  context.fillText("Centroid", legendX + 18, legendY + 4);
  if (plotData.new_point) {
    legendY += 29;
    drawNewPoint(context, legendX + 6, legendY);
    context.fillStyle = "#1f1633";
    context.fillText("Data baru", legendX + 25, legendY + 4);
  }
}

function renderPlotLegend(targetId, plotData, includeNewPoint = false) {
  const items = plotData.centroids.map((centroid) => `
    <span>
      <i style="background:${clusterColors[centroid.category]}"></i>
      ${escapeHtml(centroid.category)}
    </span>
  `);
  items.push("<span><i class=\"centroid-key\">×</i> Centroid</span>");
  if (includeNewPoint) {
    items.push("<span><i class=\"new-point-key\"></i> Data baru</span>");
  }
  document.querySelector(`#${targetId}`).innerHTML = items.join("");
}

function renderBaselinePlot(plotData) {
  const explained = plotData.explained_variance_ratio
    .reduce((total, value) => total + value, 0);
  document.querySelector("#variance-summary").textContent =
    `PC1 + PC2 menjelaskan ${formatPercent(explained)} variasi data`;
  document.querySelector("#plot-note").textContent = plotData.note;
  drawClusterPlot(
    document.querySelector("#baseline-cluster-plot"),
    plotData,
    "Clustering Dataset Lama dan Posisi Centroid",
  );
  renderPlotLegend("baseline-legend", plotData);
}

function renderPredictionPlot(plotData) {
  document.querySelector("#new-point-summary").textContent =
    `Data baru diproyeksikan ke ${plotData.new_point.category}`;
  drawClusterPlot(
    document.querySelector("#prediction-cluster-plot"),
    plotData,
    "Posisi Data Baru pada Hasil Clustering",
  );
  renderPlotLegend("prediction-legend", plotData, true);
}

function downloadPlot(plotType) {
  const canvasId = plotType === "baseline"
    ? "baseline-cluster-plot"
    : "prediction-cluster-plot";
  const filename = plotType === "baseline"
    ? "plot-dataset-lama-dan-centroid.png"
    : "plot-data-baru-dan-cluster.png";
  const link = document.createElement("a");
  link.download = filename;
  link.href = document.querySelector(`#${canvasId}`).toDataURL("image/png");
  link.click();
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
  renderPredictionPlot(result.process.cluster_plot);

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
    showStep(2);
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
    if (predictionResult || Number(item.dataset.stepTarget) <= 1) {
      showStep(Number(item.dataset.stepTarget));
    }
  });
});

document.querySelectorAll("[data-download-plot]").forEach((button) => {
  button.addEventListener("click", () => {
    downloadPlot(button.dataset.downloadPlot);
  });
});

document.querySelector("#restart-button").addEventListener("click", () => {
  predictionResult = null;
  document.querySelector("#confidence-bar").style.width = "0%";
  clearErrors();
  showStep(0);
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
