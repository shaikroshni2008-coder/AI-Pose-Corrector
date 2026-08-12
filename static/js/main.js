/**
 * AI Physiotherapist - Main Client JS Controller
 * Manages real-time telemetry polling, exercise mode switching,
 * speech synthesis voice coaching, and session metrics.
 */

let activeExerciseId = "left_biceps_curl";
let isVoiceEnabled = true;
let telemetryInterval = null;
let lastSpokenFeedback = "";
let lastRepCount = 0;
let sessionStartTime = null;
let timerInterval = null;

// Client-side camera variables
let clientStream = null;
let clientCaptureInterval = null;
let isClientSideWebcam = false;

// Client-side telemetry tracking variables
let lastFormScore = 100.0;
let lastWarnings = [];
let lastActiveDurationSeconds = 0;
let lastMinAngle = 999.0;
let lastMaxAngle = 0.0;

document.addEventListener("DOMContentLoaded", () => {
  console.log("AI Physiotherapist initialized.");
  initWebcam();
  startTimer();
});

// -------------------------------------------------------------------
// WEBCAM CAPTURE & TELEMETRY POLLING
// -------------------------------------------------------------------

async function initWebcam() {
  const video = document.getElementById("client-video");
  const liveStreamImg = document.getElementById("live-stream");
  const errorOverlay = document.getElementById("camera-error-overlay");

  if (errorOverlay) errorOverlay.style.display = "none";

  // Try requesting browser webcam access
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    try {
      clientStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" }
      });
      if (video) {
        video.srcObject = clientStream;
        video.onloadedmetadata = () => {
          video.play();
          isClientSideWebcam = true;
          console.log("Client-side webcam active.");
          startClientCaptureLoop();
        };
        return;
      }
    } catch (err) {
      console.warn("Client-side webcam permission denied or failed:", err);
      handleWebcamError(err);
    }
  } else {
    console.warn("Browser does not support getUserMedia.");
    handleWebcamError(new Error("getUserMedia is not supported by this browser."));
  }
}

function handleWebcamError(err) {
  const liveStreamImg = document.getElementById("live-stream");
  const errorOverlay = document.getElementById("camera-error-overlay");
  const errorMessage = document.getElementById("camera-error-message");

  // Fallback check: if we are running locally, we can fall back to server-side camera
  const isLocalhost = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
  
  if (isLocalhost) {
    console.log("Running locally. Falling back to server-side video feed.");
    isClientSideWebcam = false;
    if (liveStreamImg) {
      liveStreamImg.src = "/video_feed";
    }
    startTelemetryPolling();
  } else {
    // On deployed server (Render), server-side feed is impossible. Show error overlay.
    isClientSideWebcam = false;
    if (errorOverlay && errorMessage) {
      errorOverlay.style.display = "flex";
      
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        errorMessage.textContent = "Camera access denied. Please click the camera icon in your browser's address bar, allow permissions, and refresh the page.";
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        errorMessage.textContent = "No camera hardware detected. Please connect a webcam and reload the page.";
      } else {
        errorMessage.textContent = `Could not access webcam: ${err.message || err}. Please ensure no other app is using your camera and reload.`;
      }
    }
  }
}

function startClientCaptureLoop() {
  if (clientCaptureInterval) clearInterval(clientCaptureInterval);
  // Post frame at 150ms intervals (~6.6 FPS) to get real-time overlays & telemetry
  clientCaptureInterval = setInterval(captureAndProcessFrame, 150);
}

async function captureAndProcessFrame() {
  const video = document.getElementById("client-video");
  const canvas = document.getElementById("client-canvas");
  const liveStreamImg = document.getElementById("live-stream");

  if (!video || !canvas || !liveStreamImg || video.paused || video.ended) return;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  try {
    const base64Data = canvas.toDataURL("image/jpeg", 0.6); // 0.6 quality for lower network payload

    const response = await fetch("/api/process_frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: base64Data })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP status ${response.status}`);
    }

    const res = await response.json();
    if (res.status === "success" && res.data) {
      if (res.data.annotated_image) {
        liveStreamImg.src = res.data.annotated_image;
      }
      updateUI(res.data);
    }
  } catch (err) {
    console.error("Client-side frame processing error:", err);
  }
}

function startTelemetryPolling() {
  if (telemetryInterval) clearInterval(telemetryInterval);
  telemetryInterval = setInterval(fetchCurrentTelemetry, 250); // Poll 4x/sec
}

async function fetchCurrentTelemetry() {
  try {
    const response = await fetch('/api/session/current');
    const data = await response.json();

    if (data.status === 'success') {
      updateUI(data);
    }
  } catch (err) {
    console.warn("Telemetry fetch warning:", err);
  }
}

function updateUI(data) {
  // Update Repetition Count & Progress Bar
  const repCountEl = document.getElementById("rep-count-val");
  const repTargetEl = document.getElementById("rep-target-val");
  const progressBarEl = document.getElementById("rep-progress-bar");
  const progressPercentEl = document.getElementById("progress-percent-label");

  if (repCountEl) repCountEl.textContent = data.rep_count;
  if (repTargetEl) repTargetEl.textContent = data.target_reps;

  const percent = Math.min(100, Math.round((data.rep_count / data.target_reps) * 100));
  if (progressBarEl) progressBarEl.style.width = `${percent}%`;
  if (progressPercentEl) progressPercentEl.textContent = `${percent}% of target set achieved`;

  // Voice Feedback Trigger on New Rep Completed
  if (data.rep_count > lastRepCount) {
    playRepCompleteSound();
    speakVoiceFeedback(`Good job! Repetition ${data.rep_count} completed.`);
    lastRepCount = data.rep_count;

    // Pulse animation on rep counter
    if (repCountEl) {
      repCountEl.style.transform = "scale(1.25)";
      repCountEl.style.color = "var(--success)";
      setTimeout(() => {
        repCountEl.style.transform = "scale(1)";
        repCountEl.style.color = "var(--primary)";
      }, 400);
    }
  }

  // Update client-side telemetry tracking variables
  if (data.rep_count !== undefined) lastRepCount = data.rep_count;
  if (data.form_score !== undefined) lastFormScore = data.form_score;
  if (data.warnings !== undefined) lastWarnings = data.warnings;
  if (data.active_duration_seconds !== undefined) lastActiveDurationSeconds = data.active_duration_seconds;
  if (data.min_angle_achieved !== undefined) lastMinAngle = data.min_angle_achieved;
  if (data.max_angle_achieved !== undefined) lastMaxAngle = data.max_angle_achieved;

  // Update Live Angle Gauge
  const angleValEl = document.getElementById("angle-val");
  if (angleValEl) angleValEl.textContent = Math.round(data.current_angle);

  // Update Form Score & Badge
  const scorePillEl = document.getElementById("form-score-pill");
  if (scorePillEl) scorePillEl.textContent = `Form: ${Math.round(data.form_score)}%`;

  const stageBadgeEl = document.getElementById("stage-badge");
  if (stageBadgeEl) {
    stageBadgeEl.textContent = data.stage;
    if (data.stage === "FLEXED") {
      stageBadgeEl.className = "score-badge good";
    } else if (data.stage === "EXTENDED") {
      stageBadgeEl.className = "score-badge warning";
    } else {
      stageBadgeEl.className = "score-badge good";
    }
  }

  // Update Live Feedback Banner
  const feedbackBannerEl = document.getElementById("live-feedback-banner");
  if (feedbackBannerEl && data.feedback) {
    feedbackBannerEl.innerHTML = `<i class="bi bi-info-circle-fill"></i> ${data.feedback}`;
    
    // Voice coaching for state changes
    if (isVoiceEnabled && data.feedback !== lastSpokenFeedback && data.feedback.length > 5) {
      speakVoiceFeedback(data.feedback);
      lastSpokenFeedback = data.feedback;
    }
  }

  // Update Form Warnings Box
  const warningsBoxEl = document.getElementById("warnings-box");
  if (warningsBoxEl) {
    if (data.warnings && data.warnings.length > 0) {
      warningsBoxEl.innerHTML = data.warnings.map(w => `
        <div style="display: flex; align-items: center; gap: 8px; color: var(--warning); margin-bottom: 4px;">
          <i class="bi bi-exclamation-triangle-fill"></i> ${w}
        </div>
      `).join('');
    } else {
      warningsBoxEl.innerHTML = `
        <div style="display: flex; align-items: center; gap: 8px; color: var(--success);">
          <i class="bi bi-check-circle-fill"></i> Posture aligned correctly.
        </div>
      `;
    }
  }
}

// -------------------------------------------------------------------
// EXERCISE MODE SWITCHING
// -------------------------------------------------------------------

async function switchExercise(exerciseId) {
  try {
    const response = await fetch('/api/exercise/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ exercise_id: exerciseId })
    });
    const res = await response.json();

    if (res.status === 'success') {
      activeExerciseId = exerciseId;
      lastRepCount = 0;

      // Update sidebar UI card selection
      document.querySelectorAll('.exercise-card').forEach(card => card.classList.remove('active'));
      const activeCard = document.getElementById(`ex-card-${exerciseId}`);
      if (activeCard) activeCard.classList.add('active');

      // Update Header & Angle Labels
      const ex = res.exercise;
      document.getElementById("active-ex-name").textContent = ex.name;
      document.getElementById("target-angle-label").textContent = `${ex.angle_target}°`;
      document.getElementById("rest-angle-label").textContent = `${ex.angle_rest}°`;
      document.getElementById("rep-target-val").textContent = ex.recommended_reps;

      // Update Tips List
      const tipsContainer = document.getElementById("tips-container");
      if (tipsContainer && ex.posture_tips) {
        tipsContainer.innerHTML = ex.posture_tips.map(tip => `
          <li><i class="bi bi-check-circle-fill"></i> ${tip}</li>
        `).join('');
      }

      showToast(`Switched mode to ${ex.name}`);
      speakVoiceFeedback(`Switched to ${ex.name}`);
      resetTimer();
    }
  } catch (err) {
    console.error("Exercise switch error:", err);
  }
}

// -------------------------------------------------------------------
// SESSION CONTROLS
// -------------------------------------------------------------------

async function startSession() {
  try {
    const res = await fetch('/api/session/start', { method: 'POST' });
    const data = await res.json();
    lastRepCount = 0;
    resetTimer();
    showToast("New exercise set started!");
    speakVoiceFeedback("Starting new set. Begin motion when ready.");
  } catch (err) {
    console.error("Start session error:", err);
  }
}

async function stopSession() {
  try {
    const res = await fetch('/api/session/stop', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        reps: lastRepCount,
        accuracy: lastFormScore,
        warnings: lastWarnings,
        duration_seconds: lastActiveDurationSeconds,
        min_angle: lastMinAngle,
        max_angle: lastMaxAngle
      })
    });
    const data = await res.json();

    if (data.status === 'success') {
      showToast(`Session saved! ${data.summary.reps_completed} reps completed.`);
      speakVoiceFeedback(`Set completed! Saved ${data.summary.reps_completed} repetitions.`);
    }
  } catch (err) {
    console.error("Stop session error:", err);
  }
}

// -------------------------------------------------------------------
// SPEECH SYNTHESIS VOICE COACH
// -------------------------------------------------------------------

function toggleVoiceCoaching() {
  isVoiceEnabled = !isVoiceEnabled;
  const btn = document.getElementById("voice-toggle-btn");
  if (btn) {
    if (isVoiceEnabled) {
      btn.innerHTML = `<i class="bi bi-volume-up-fill"></i> Voice Coach: ON`;
      btn.classList.remove('btn-danger');
      btn.classList.add('btn-secondary');
      speakVoiceFeedback("Voice coaching enabled");
    } else {
      btn.innerHTML = `<i class="bi bi-volume-mute-fill"></i> Voice Coach: OFF`;
      btn.classList.remove('btn-secondary');
      btn.classList.add('btn-danger');
    }
  }
}

function speakVoiceFeedback(text) {
  if (!isVoiceEnabled || !('speechSynthesis' in window)) return;
  
  // Cancel previous speech if needed
  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.0;
  utterance.pitch = 1.05;
  utterance.volume = 0.9;
  window.speechSynthesis.speak(utterance);
}

function playRepCompleteSound() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(587.33, ctx.currentTime); // D5 note
    osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.15); // A5 note
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.25);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.25);
  } catch (e) {
    // Ignore audio context autoplay restrictions
  }
}

// -------------------------------------------------------------------
// TIMER & TOAST UTILITIES
// -------------------------------------------------------------------

function startTimer() {
  sessionStartTime = Date.now();
  if (timerInterval) clearInterval(timerInterval);
  timerInterval = setInterval(updateTimerDisplay, 1000);
}

function resetTimer() {
  sessionStartTime = Date.now();
  updateTimerDisplay();
}

function updateTimerDisplay() {
  if (!sessionStartTime) return;
  const elapsedSec = Math.floor((Date.now() - sessionStartTime) / 1000);
  const mins = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
  const secs = String(elapsedSec % 60).padStart(2, '0');
  const timerEl = document.getElementById("timer-display");
  if (timerEl) timerEl.textContent = `${mins}:${secs}`;
}

function showToast(message) {
  const toast = document.getElementById("toast");
  const msgEl = document.getElementById("toast-msg");
  if (toast && msgEl) {
    msgEl.textContent = message;
    toast.classList.add("show");
    setTimeout(() => {
      toast.classList.remove("show");
    }, 3500);
  }
}
