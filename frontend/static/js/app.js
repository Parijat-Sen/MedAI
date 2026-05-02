/**
 * ============================================================
 * MedAI — Frontend Application Logic
 * ============================================================
 * Handles:
 *   - Symptom analysis flow
 *   - API calls to FastAPI backend
 *   - Chat interface
 *   - Metrics display
 *   - UI state management
 * ============================================================
 */

const API_BASE = "";  // Same origin as backend serves frontend

// ── App state ─────────────────────────────────────────────
let currentSessionId = null;
let currentAnalysisContext = "";
let isAnalyzing = false;

// ══════════════════════════════════════════════════════════
// INITIALIZATION
// ══════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  checkHealth();
  loadMetrics();
  setInterval(checkHealth, 30000); // Health check every 30s
});

// ══════════════════════════════════════════════════════════
// NAVIGATION
// ══════════════════════════════════════════════════════════

function setupNavigation() {
  document.querySelectorAll(".nav-link").forEach(link => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const viewName = link.dataset.view;
      switchView(viewName);
    });
  });
}

function switchView(viewName) {
  // Update nav links
  document.querySelectorAll(".nav-link").forEach(l => l.classList.remove("active"));
  document.querySelector(`[data-view="${viewName}"]`)?.classList.add("active");

  // Update views
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  document.getElementById(`view-${viewName}`)?.classList.add("active");

  // Load metrics when switching to that tab
  if (viewName === "metrics") loadMetrics();
}

// ══════════════════════════════════════════════════════════
// HEALTH CHECK
// ══════════════════════════════════════════════════════════

async function checkHealth() {
  const badge = document.getElementById("statusBadge");
  const dot = badge.querySelector(".status-dot");
  const text = document.getElementById("statusText");

  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();

    if (data.status === "healthy") {
      dot.className = "status-dot online";
      const mlOk = data.components.ml_model;
      const ragOk = data.components.rag_pipeline;
      const provider = data.components.llm_provider;
      text.textContent = `ML:${mlOk ? "✓" : "✗"} RAG:${ragOk ? "✓" : "✗"} LLM:${provider}`;
    }
  } catch {
    dot.className = "status-dot offline";
    text.textContent = "Offline";
  }
}

// ══════════════════════════════════════════════════════════
// QUICK SYMPTOM ADD
// ══════════════════════════════════════════════════════════

function addQuickSymptom(symptom) {
  const textarea = document.getElementById("symptomText");
  const current = textarea.value.trim();

  if (!current.toLowerCase().includes(symptom.toLowerCase())) {
    textarea.value = current
      ? `${current}, ${symptom}`
      : symptom;
  }
}

// ══════════════════════════════════════════════════════════
// MAIN ANALYSIS FLOW
// ══════════════════════════════════════════════════════════

async function analyzeSymptoms() {
  const text = document.getElementById("symptomText").value.trim();

  if (!text) {
    showError("Please describe your symptoms first.");
    return;
  }

  if (text.length < 5) {
    showError("Please provide more detail about your symptoms.");
    return;
  }

  if (isAnalyzing) return;

  // Generate a session ID for this analysis
  currentSessionId = `session_${Date.now()}`;

  // Show loading state
  setLoadingState(true);
  isAnalyzing = true;

  try {
    const response = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text,
        session_id: currentSessionId,
        top_n: 3
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `Server error (${response.status})`);
    }

    const data = await response.json();
    renderResults(data);

  } catch (err) {
    console.error("Analysis failed:", err);
    showError(err.message || "Analysis failed. Make sure the backend is running.");
  } finally {
    setLoadingState(false);
    isAnalyzing = false;
  }
}

// ── Loading state animation ────────────────────────────────
function setLoadingState(loading) {
  const btn = document.getElementById("analyzeBtn");
  const loadingEl = document.getElementById("loadingState");
  const emptyEl = document.getElementById("emptyState");
  const resultsEl = document.getElementById("resultsContent");
  const errorEl = document.getElementById("errorState");

  if (loading) {
    btn.disabled = true;
    btn.innerHTML = '<span class="loading-spinner" style="width:16px;height:16px;border-width:2px;"></span> Analyzing...';
    loadingEl.style.display = "flex";
    emptyEl.style.display = "none";
    resultsEl.style.display = "none";
    errorEl.style.display = "none";

    // Animate loading steps
    const steps = ["ls1", "ls2", "ls3", "ls4"];
    steps.forEach(id => document.getElementById(id).className = "loading-step");
    document.getElementById("ls1").className = "loading-step active";

    let step = 0;
    const interval = setInterval(() => {
      if (!isAnalyzing) { clearInterval(interval); return; }
      document.getElementById(steps[step]).className = "loading-step done";
      step++;
      if (step < steps.length) {
        document.getElementById(steps[step]).className = "loading-step active";
      } else {
        clearInterval(interval);
      }
    }, 1200);

  } else {
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">🔬</span> Analyze Symptoms';
    loadingEl.style.display = "none";
  }
}

// ══════════════════════════════════════════════════════════
// RENDER RESULTS
// ══════════════════════════════════════════════════════════

function renderResults(data) {
  // Show results content
  document.getElementById("emptyState").style.display = "none";
  document.getElementById("errorState").style.display = "none";
  document.getElementById("resultsContent").style.display = "flex";

  // Processing time badge
  const timeBadge = document.getElementById("processingTime");
  timeBadge.textContent = `${data.elapsed_seconds}s`;
  timeBadge.style.display = "inline-flex";

  // Render detected symptom tags
  renderSymptomTags(data.input.extracted_symptoms);

  // Render ML predictions
  renderPredictions(data.ml_predictions);

  // Render LLM analysis
  const llmEl = document.getElementById("llmExplanation");
  llmEl.textContent = data.llm_analysis.explanation;

  // Red flags
  const redFlagsSection = document.getElementById("redFlagsSection");
  const redFlagsEl = document.getElementById("redFlags");
  const redFlags = data.llm_analysis.red_flags || [];

  if (redFlags.length > 0) {
    redFlagsSection.style.display = "block";
    redFlagsEl.innerHTML = redFlags.map(flag => `
      <div class="red-flag-item">
        <span>🚨</span>
        <span>${escapeHtml(flag)}</span>
      </div>
    `).join("");
  } else {
    redFlagsSection.style.display = "none";
  }

  // Follow-up questions
  const followUpEl = document.getElementById("followUpQuestions");
  const questions = data.llm_analysis.follow_up_questions || [];
  followUpEl.innerHTML = questions.map((q, i) => `
    <div class="followup-item">
      <div class="item-num">${i + 1}</div>
      <span>${escapeHtml(q)}</span>
    </div>
  `).join("");

  // Recommended actions
  const actionsEl = document.getElementById("recommendedActions");
  const actions = data.llm_analysis.recommended_actions || [];
  actionsEl.innerHTML = actions.map((a, i) => `
    <div class="action-item">
      <div class="item-num">${i + 1}</div>
      <span>${escapeHtml(a)}</span>
    </div>
  `).join("");

  // Disclaimer
  document.getElementById("disclaimerText").textContent = data.llm_analysis.disclaimer;

  // Store context for chat
  currentAnalysisContext = data.llm_analysis.explanation;

  // Scroll to results
  document.getElementById("resultsPanel").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderSymptomTags(symptoms) {
  const container = document.getElementById("symptomTags");
  if (!symptoms || symptoms.length === 0) {
    container.innerHTML = '<span class="tag-placeholder">No recognized symptoms extracted.</span>';
    return;
  }

  container.innerHTML = symptoms.map(s => `
    <div class="symptom-tag">
      <span>${s.replace(/_/g, " ")}</span>
      <span class="tag-remove" onclick="removeSymptomTag('${s}')">×</span>
    </div>
  `).join("");
}

function removeSymptomTag(symptom) {
  const textarea = document.getElementById("symptomText");
  const readable = symptom.replace(/_/g, " ");
  // Remove symptom from textarea
  textarea.value = textarea.value
    .replace(new RegExp(`,?\\s*${readable}\\s*,?`, "gi"), ",")
    .replace(/^,\s*/, "")
    .replace(/,\s*$/, "")
    .trim();
}

function renderPredictions(predictions) {
  const container = document.getElementById("predictionsGrid");

  if (!predictions || predictions.length === 0) {
    container.innerHTML = '<div class="empty-text">No predictions available.</div>';
    return;
  }

  container.innerHTML = predictions.map((p, i) => `
    <div class="prediction-card ${i === 0 ? 'rank-1' : ''}">
      <div>
        <div class="pred-rank">#${p.rank} Prediction</div>
        <div class="pred-disease">${escapeHtml(p.disease)}</div>
        <div class="pred-meta">
          <span class="pred-tag ${p.severity}">${p.severity || "unknown"}</span>
          <span class="pred-tag">${escapeHtml(p.specialist || "GP")}</span>
          ${i === 0 ? '<span class="pred-tag" style="background:rgba(0,180,166,0.1);color:var(--teal)">Top Match</span>' : ""}
        </div>
        <div style="margin-top:8px;font-size:12px;color:var(--text-muted);">${escapeHtml(p.urgency || "")}</div>
      </div>
      <div class="pred-confidence">
        <div class="conf-value">${p.confidence.toFixed(1)}%</div>
        <div class="conf-bar">
          <div class="conf-bar-fill" style="width:${p.confidence}%"></div>
        </div>
        <div class="conf-level">${p.confidence_level}</div>
      </div>
    </div>
  `).join("");
}

// ══════════════════════════════════════════════════════════
// ERROR HANDLING
// ══════════════════════════════════════════════════════════

function showError(message) {
  document.getElementById("emptyState").style.display = "none";
  document.getElementById("resultsContent").style.display = "none";
  document.getElementById("loadingState").style.display = "none";
  document.getElementById("errorState").style.display = "flex";
  document.getElementById("errorMessage").textContent = message;
}

function clearError() {
  document.getElementById("errorState").style.display = "none";
  document.getElementById("emptyState").style.display = "flex";
}

function clearAll() {
  document.getElementById("symptomText").value = "";
  document.getElementById("symptomTags").innerHTML = '<span class="tag-placeholder">Symptoms will appear here after analysis...</span>';
  document.getElementById("emptyState").style.display = "flex";
  document.getElementById("resultsContent").style.display = "none";
  document.getElementById("errorState").style.display = "none";
  document.getElementById("loadingState").style.display = "none";
  document.getElementById("processingTime").style.display = "none";
  currentSessionId = null;
}

// ══════════════════════════════════════════════════════════
// CHAT
// ══════════════════════════════════════════════════════════

function startChat() {
  if (!currentSessionId) {
    alert("Please run an analysis first.");
    return;
  }

  // Switch to chat view
  switchView("chat");

  // Enable chat
  const input = document.getElementById("chatInput");
  const btn = document.getElementById("chatSendBtn");
  input.disabled = false;
  btn.disabled = false;

  // Update session info
  document.getElementById("chatSessionInfo").textContent =
    `Session: ${currentSessionId} | Analysis complete — ask follow-up questions`;

  // Show context message
  appendChatMessage(
    "assistant",
    `I've completed your symptom analysis. Based on the results, do you have any follow-up questions? You can ask me to clarify predictions, explain medical terms, or provide more information about any of the suggested conditions.`
  );
}

async function sendChat() {
  const input = document.getElementById("chatInput");
  const message = input.value.trim();

  if (!message) return;

  if (!currentSessionId) {
    appendChatMessage("assistant", "Please run a symptom analysis first from the Analyze tab.");
    return;
  }

  // Show user message
  appendChatMessage("user", message);
  input.value = "";
  input.disabled = true;
  document.getElementById("chatSendBtn").disabled = true;

  // Typing indicator
  const typingId = appendChatMessage("assistant", "...", true);

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: message,
        session_id: currentSessionId
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Chat request failed");
    }

    const data = await res.json();
    const reply = data.response.explanation || "I couldn't generate a response. Please try again.";

    // Replace typing indicator
    removeMessage(typingId);
    appendChatMessage("assistant", reply);

    // Show follow-ups if any
    const questions = data.response.follow_up_questions || [];
    if (questions.length > 0) {
      const qText = "I might also ask:\n" + questions.slice(0, 2).map((q, i) => `${i+1}. ${q}`).join("\n");
      appendChatMessage("assistant", qText);
    }

  } catch (err) {
    removeMessage(typingId);
    appendChatMessage("assistant", `Error: ${err.message}`);
  } finally {
    input.disabled = false;
    document.getElementById("chatSendBtn").disabled = false;
    input.focus();
  }
}

function appendChatMessage(role, text, isTyping = false) {
  const container = document.getElementById("chatMessages");
  const id = `msg_${Date.now()}_${Math.random().toString(36).substr(2,5)}`;

  const div = document.createElement("div");
  div.className = `chat-message ${role}`;
  div.id = id;

  const avatar = role === "assistant" ? "⚕" : "👤";
  div.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-bubble" ${isTyping ? 'style="font-style:italic;opacity:0.6"' : ''}>
      ${escapeHtml(text).replace(/\n/g, "<br>")}
    </div>
  `;

  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return id;
}

function removeMessage(id) {
  document.getElementById(id)?.remove();
}

// ══════════════════════════════════════════════════════════
// METRICS
// ══════════════════════════════════════════════════════════

async function loadMetrics() {
  const container = document.getElementById("metricsContent");

  try {
    const res = await fetch(`${API_BASE}/api/metrics`);
    if (!res.ok) throw new Error("Metrics not available");

    const data = await res.json();
    renderMetrics(data, container);
  } catch (err) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">📊</div>
        <div class="empty-title">Metrics Unavailable</div>
        <div class="empty-text">Train the ML model first: <code>python setup.py</code><br><br>Error: ${err.message}</div>
      </div>
    `;
  }
}

function renderMetrics(data, container) {
  const metrics = data.metrics || {};
  const bestModel = data.best_model || "Unknown";
  const topFeatures = data.top_features || [];
  const info = data.training_info || {};

  let html = `
    <div style="margin-bottom:8px;">
      <span class="section-label">Best Model: <span style="color:var(--teal);font-size:14px;">${escapeHtml(bestModel)}</span></span>
    </div>

    <div class="metrics-grid">
  `;

  // Render metric cards for each model
  for (const [model, m] of Object.entries(metrics)) {
    const isBest = model === bestModel;
    html += `
      <div class="metric-card" style="${isBest ? 'border-color:var(--teal)' : ''}">
        ${isBest ? '<div style="font-size:11px;color:var(--teal);margin-bottom:4px;font-family:var(--font-mono)">★ BEST</div>' : ''}
        <div class="metric-model">${escapeHtml(model)}</div>
        <div class="metric-value">${(m.accuracy * 100).toFixed(1)}%</div>
        <div class="metric-label">Accuracy</div>
        <div class="metric-details">
          <span>F1 (macro): ${(m.f1_macro * 100).toFixed(1)}%</span>
          <span>Precision: ${(m.precision_macro * 100).toFixed(1)}%</span>
          <span>Recall: ${(m.recall_macro * 100).toFixed(1)}%</span>
          <span>CV F1: ${m.cv_f1_macro ? (m.cv_f1_macro * 100).toFixed(1) + '%' : 'N/A'}</span>
        </div>
      </div>
    `;
  }

  html += `</div>`;

  // Training info
  html += `
    <div class="feature-importance">
      <h3>Training Summary</h3>
      <div class="metric-details" style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:12px;">
        <span>Diseases: <strong style="color:var(--teal)">${info.num_classes || '?'}</strong></span>
        <span>Symptoms: <strong style="color:var(--teal)">${info.num_features || '?'}</strong></span>
        <span>Samples: <strong style="color:var(--teal)">${info.training_samples || '?'}</strong></span>
        <span>Classes: <strong style="color:var(--teal)">${(info.class_names || []).length}</strong></span>
      </div>
    </div>
  `;

  // Feature importance
  if (topFeatures.length > 0) {
    const maxImportance = topFeatures[0].importance;
    html += `
      <div class="feature-importance">
        <h3>Top Predictive Symptoms (Feature Importance)</h3>
        ${topFeatures.slice(0, 15).map(f => `
          <div class="feature-bar">
            <div class="feature-name">${f.symptom.replace(/_/g, " ")}</div>
            <div class="feature-track">
              <div class="feature-fill" style="width:${(f.importance / maxImportance * 100).toFixed(1)}%"></div>
            </div>
            <div class="feature-pct">${(f.importance * 100).toFixed(2)}%</div>
          </div>
        `).join("")}
      </div>
    `;
  }

  container.innerHTML = html;
}

// ══════════════════════════════════════════════════════════
// UTILITIES
// ══════════════════════════════════════════════════════════

function escapeHtml(text) {
  if (!text) return "";
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// Keyboard shortcut: Ctrl/Cmd + Enter to analyze
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    const activeView = document.querySelector(".view.active");
    if (activeView?.id === "view-analyze") {
      analyzeSymptoms();
    }
  }
});