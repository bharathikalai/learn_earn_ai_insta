const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const dropzoneText = document.getElementById("dropzone-text");
const fileNameEl = document.getElementById("fileName");
const statusPill = document.getElementById("status-pill");

const playButton = document.getElementById("playButton");
const stopButton = document.getElementById("stopButton");

const predictionLabel = document.getElementById("predictionLabel");
const confidenceText = document.getElementById("confidenceText");
const probabilitiesContainer = document.getElementById("probabilities");

const waveformCanvas = document.getElementById("waveformCanvas");
const canvasCtx = waveformCanvas.getContext("2d");

let currentFile = null;
let audioCtx = null;
let audioBuffer = null;
let sourceNode = null;

// Resize canvas to actual pixel size
function resizeCanvas() {
  const rect = waveformCanvas.getBoundingClientRect();
  waveformCanvas.width = rect.width;
  waveformCanvas.height = rect.height;
}
window.addEventListener("resize", resizeCanvas);
resizeCanvas();

// ------------- FILE HANDLING -------------
dropzone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) handleFile(file);
});

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) {
    fileInput.files = e.dataTransfer.files;
    handleFile(file);
  }
});

function handleFile(file) {
  currentFile = file;
  fileNameEl.textContent = file.name;
  dropzoneText.textContent = "File loaded. Click to change file.";
  playButton.disabled = false;
  stopButton.disabled = false;

  resetPredictionUI("Analyzing audio…");
  drawWaveform(file);
  sendToBackend(file);
}

// ------------- WAVEFORM DRAWING -------------
function drawWaveform(file) {
  if (!window.AudioContext && !window.webkitAudioContext) {
    console.warn("Web Audio API not supported in this browser.");
    return;
  }

  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }

  const reader = new FileReader();
  reader.onload = (event) => {
    const arrayBuffer = event.target.result;
    audioCtx.decodeAudioData(arrayBuffer)
      .then((buffer) => {
        audioBuffer = buffer;
        renderWaveform(buffer);
      })
      .catch((err) => console.error("decodeAudioData error:", err));
  };
  reader.readAsArrayBuffer(file);
}

function renderWaveform(buffer) {
  const width = waveformCanvas.width;
  const height = waveformCanvas.height;
  const data = buffer.getChannelData(0);

  canvasCtx.clearRect(0, 0, width, height);

  // Background
  canvasCtx.fillStyle = "rgba(0, 0, 0, 0.9)";
  canvasCtx.fillRect(0, 0, width, height);

  // Midline
  const midY = height / 2;
  canvasCtx.strokeStyle = "rgba(255, 255, 255, 0.2)";
  canvasCtx.beginPath();
  canvasCtx.moveTo(0, midY);
  canvasCtx.lineTo(width, midY);
  canvasCtx.stroke();

  const samples = 1000;
  const blockSize = Math.floor(data.length / samples);
  const amplitudes = [];

  for (let i = 0; i < samples; i++) {
    let sum = 0;
    for (let j = 0; j < blockSize; j++) {
      sum += Math.abs(data[i * blockSize + j]);
    }
    amplitudes.push(sum / blockSize);
  }

  const maxAmp = Math.max(...amplitudes) || 1;

  canvasCtx.lineWidth = 2;
  canvasCtx.strokeStyle = "#00ff99";
  canvasCtx.beginPath();

  for (let i = 0; i < samples; i++) {
    const x = (i / (samples - 1)) * width;
    const amp = amplitudes[i] / maxAmp;
    const y = midY - amp * (height / 2 - 6);
    if (i === 0) {
      canvasCtx.moveTo(x, y);
    } else {
      canvasCtx.lineTo(x, y);
    }
  }

  canvasCtx.stroke();
}

// ------------- AUDIO PLAYBACK -------------
playButton.addEventListener("click", () => {
  if (!audioBuffer) return;
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }

  if (sourceNode) {
    sourceNode.stop();
    sourceNode.disconnect();
  }

  sourceNode = audioCtx.createBufferSource();
  sourceNode.buffer = audioBuffer;
  sourceNode.connect(audioCtx.destination);
  sourceNode.start(0);

  statusPill.textContent = "Playing";
});

stopButton.addEventListener("click", () => {
  if (sourceNode) {
    sourceNode.stop();
    sourceNode.disconnect();
    sourceNode = null;
  }
  statusPill.textContent = "Idle";
});

// ------------- BACKEND COMMUNICATION -------------
function sendToBackend(file) {
  statusPill.textContent = "Analyzing…";

  const formData = new FormData();
  formData.append("audio", file);

  fetch("/predict", {
    method: "POST",
    body: formData
  })
    .then((res) => res.json())
    .then((data) => {
      if (data.error) {
        resetPredictionUI("Error: " + data.error);
        statusPill.textContent = "Error";
        return;
      }

      updatePredictionUI(data);
      statusPill.textContent = "Done";
    })
    .catch((err) => {
      console.error(err);
      resetPredictionUI("Error talking to server.");
      statusPill.textContent = "Error";
    });
}

function resetPredictionUI(message) {
  predictionLabel.textContent = "—";
  confidenceText.textContent = message || "Upload audio to analyze.";
  probabilitiesContainer.innerHTML = "";
}

function updatePredictionUI(data) {
  const label = data.label || "unknown";
  const probs = data.probabilities || {};

  predictionLabel.textContent = label.toUpperCase();

  // Find top probability
  let maxClass = null;
  let maxProb = -1;
  for (const [cls, p] of Object.entries(probs)) {
    if (p > maxProb) {
      maxProb = p;
      maxClass = cls;
    }
  }
  const pct = (maxProb * 100).toFixed(1);
  confidenceText.textContent = `${maxClass.toUpperCase()} • Confidence: ${pct}%`;

  probabilitiesContainer.innerHTML = "";

  Object.entries(probs).forEach(([cls, p]) => {
    const row = document.createElement("div");
    row.className = "prob-row";

    const labelEl = document.createElement("div");
    labelEl.className = "prob-label";
    labelEl.textContent = cls.toUpperCase();

    const bar = document.createElement("div");
    bar.className = "prob-bar";

    const barFill = document.createElement("div");
    barFill.className = "prob-bar-fill";
    barFill.style.width = (p * 100).toFixed(1) + "%";

    bar.appendChild(barFill);

    const valueEl = document.createElement("div");
    valueEl.className = "prob-value";
    valueEl.textContent = (p * 100).toFixed(1) + "%";

    row.appendChild(labelEl);
    row.appendChild(bar);
    row.appendChild(valueEl);

    probabilitiesContainer.appendChild(row);
  });
}
