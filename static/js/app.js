/**
 * CardioScan AI — Drug Classification System
 * app.js — Frontend Logic
 *
 * Handles:
 *  - Page navigation with transitions
 *  - Form validation
 *  - Fetch POST to Flask /predict endpoint
 *  - Result rendering with animated probability bar
 */

/* =========================================================
   PAGE NAVIGATION
   ========================================================= */

/**
 * Switch visible page sections by ID.
 * @param {string} pageId - 'landing' | 'prediction' | 'result'
 */
function showPage(pageId) {
  document.querySelectorAll('.page').forEach(p => {
    p.classList.remove('active');
  });

  const target = document.getElementById(pageId);
  if (target) {
    target.classList.add('active');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
}

/* =========================================================
   FORM SUBMISSION
   ========================================================= */

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('predForm');
  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    await handleSubmit(form);
  });
});

/**
 * Collect form data, send to /predict, render result.
 * @param {HTMLFormElement} form
 */
async function handleSubmit(form) {
  const submitBtn = document.getElementById('submitBtn');

  // --- Read raw form values ---
  const age        = parseFloat(document.getElementById('age').value);
  const gender     = parseFloat(document.getElementById('gender').value);
  const height     = parseFloat(document.getElementById('height').value);
  const weight     = parseFloat(document.getElementById('weight').value);
  const systolic   = parseFloat(document.getElementById('systolic').value);
  const diastolic  = parseFloat(document.getElementById('diastolic').value);
  const cholesterol = parseFloat(document.getElementById('cholesterol').value);
  const glucose    = parseFloat(document.getElementById('glucose').value);
  const smoking    = parseFloat(document.getElementById('smoking').value);
  const alcohol    = parseFloat(document.getElementById('alcohol').value);
  const activity   = parseFloat(document.getElementById('activity').value);

  // Basic guard against NaN (HTML5 required should catch most cases)
  const fields = { age, gender, height, weight, systolic, diastolic, cholesterol, glucose, smoking, alcohol, activity };
  for (const [key, val] of Object.entries(fields)) {
    if (isNaN(val)) {
      alert(`Please fill in all fields. Missing: ${key}`);
      return;
    }
  }

  // Payload keys must match what the Flask backend expects
  const payload = {
    age:         age,
    gender:      gender,
    height:      height,
    weight:      weight,
    ap_hi:       systolic,
    ap_lo:       diastolic,
    cholesterol: cholesterol,
    gluc:        glucose,
    smoke:       smoking,
    alco:        alcohol,
    active:      activity
  };

  // --- Loading state ---
  setLoading(submitBtn, true);

  try {
    const response = await fetch('http://127.0.0.1:5001/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || `Server error: ${response.status}`);
    }

    const data = await response.json();
    // Expected: { prediction: 0|1, probability: 0.xx }

    renderResult(data, payload);
    showPage('result');

  } catch (err) {
    showErrorToast(err.message || 'Failed to connect to prediction server.');
    console.error('Prediction error:', err);
  } finally {
    setLoading(submitBtn, false);
  }
}

/* =========================================================
   RESULT RENDERING
   ========================================================= */

/**
 * Populate the result page with prediction data.
 * @param {{ prediction: number, probability: number }} data - API response
 * @param {Object} inputs - original form values (for summary)
 */
function renderResult(data, inputs) {
  const isRisk = data.prediction === 1;
  const probPct = Math.round((data.probability || 0) * 100);

  const card        = document.getElementById('resultCard');
  const badge       = document.getElementById('resultBadge');
  const icon        = document.getElementById('resultIcon');
  const title       = document.getElementById('resultTitle');
  const subtitle    = document.getElementById('resultSubtitle');
  const probText    = document.getElementById('probText');
  const probBar     = document.getElementById('probBar');
  const details     = document.getElementById('resultDetails');

  // Apply risk/safe class to card for color theming
  card.classList.remove('risk', 'safe');
  card.classList.add(isRisk ? 'risk' : 'safe');

  // Badge
  badge.textContent = isRisk ? '⚠ Risk Detected' : '✓ No Risk Detected';
  badge.className   = 'result-badge ' + (isRisk ? 'risk' : 'safe');

  // Icon
  icon.textContent = isRisk ? '🫀' : '💚';

  // Title & subtitle
  title.textContent = isRisk
    ? 'Cardiotoxic Risk Detected'
    : 'Non-Cardiotoxic Drug';

  subtitle.textContent = isRisk
    ? 'The AI model indicates a significant risk of drug-induced cardiotoxicity based on the provided patient biomarkers. Clinical review is recommended.'
    : 'The AI model indicates no significant cardiotoxicity risk for this patient profile. Continue standard monitoring protocols.';

  // Probability display
  probText.textContent = `${probPct}%`;

  // Animate probability bar after a short delay
  setTimeout(() => {
    probBar.style.width = `${probPct}%`;
  }, 200);

  // Details grid — summary of key inputs
  const detailItems = [
    { key: 'Age',               val: `${inputs.age} years`           },
    { key: 'Gender',            val: inputs.gender === 1 ? 'Male' : 'Female' },
    { key: 'BMI',               val: calcBMI(inputs.height, inputs.weight) },
    { key: 'Blood Pressure',    val: `${inputs.ap_hi}/${inputs.ap_lo} mmHg` },
    { key: 'Cholesterol',       val: levelLabel(inputs.cholesterol)  },
    { key: 'Glucose',           val: levelLabel(inputs.gluc)         },
    { key: 'Smoker',            val: inputs.smoke === 1 ? 'Yes' : 'No' },
    { key: 'Alcohol Use',       val: inputs.alco  === 1 ? 'Yes' : 'No' },
    { key: 'Physical Activity', val: inputs.active === 1 ? 'Active' : 'Inactive' },
    { key: 'Confidence Score',  val: `${probPct}%` }
  ];

  details.innerHTML = detailItems.map(d => `
    <div class="detail-item">
      <div class="detail-key">${d.key}</div>
      <div class="detail-val">${d.val}</div>
    </div>
  `).join('');
}

/* =========================================================
   HELPERS
   ========================================================= */

/**
 * Toggle loading state on submit button.
 */
function setLoading(btn, isLoading) {
  if (!btn) return;
  btn.disabled = isLoading;
  if (isLoading) {
    btn.classList.add('loading');
  } else {
    btn.classList.remove('loading');
  }
}

/**
 * Calculate BMI and return formatted string.
 */
function calcBMI(heightCm, weightKg) {
  if (!heightCm || !weightKg) return '—';
  const bmi = weightKg / ((heightCm / 100) ** 2);
  const label = bmi < 18.5 ? 'Underweight'
              : bmi < 25   ? 'Normal'
              : bmi < 30   ? 'Overweight'
              : 'Obese';
  return `${bmi.toFixed(1)} (${label})`;
}

/**
 * Convert numeric cholesterol/glucose level to text.
 */
function levelLabel(val) {
  if (val === 1) return 'Normal';
  if (val === 2) return 'Above Normal';
  if (val === 3) return 'Well Above Normal';
  return '—';
}

/**
 * Show a simple error toast message.
 * @param {string} message
 */
function showErrorToast(message) {
  // Remove existing toast
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.innerHTML = `
    <span style="font-size:18px;">⚠️</span>
    <span>${message}</span>
  `;
  toast.style.cssText = `
    position: fixed;
    bottom: 32px;
    left: 50%;
    transform: translateX(-50%);
    background: #1a0a10;
    border: 1px solid var(--red, #ff4d6d);
    color: #ff8fa3;
    padding: 14px 24px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 14px;
    font-family: 'DM Sans', sans-serif;
    z-index: 9999;
    box-shadow: 0 8px 32px #00000060;
    animation: toastIn 0.3s ease;
    max-width: 90vw;
  `;

  // Inject keyframe if not already present
  if (!document.getElementById('toastStyle')) {
    const style = document.createElement('style');
    style.id = 'toastStyle';
    style.textContent = `
      @keyframes toastIn {
        from { opacity:0; transform: translateX(-50%) translateY(16px); }
        to   { opacity:1; transform: translateX(-50%) translateY(0); }
      }
    `;
    document.head.appendChild(style);
  }

  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 5000);
}