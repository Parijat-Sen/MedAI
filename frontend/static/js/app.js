const API_BASE = window.location.origin;

let isAnalyzing = false;
let currentSessionId = null;

// ================= INIT =================
document.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  checkHealth();
  setInterval(checkHealth, 30000);
});

// ================= NAV =================
function setupNavigation() {
  document.querySelectorAll(".nav-link").forEach(link => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      switchView(link.dataset.view);
    });
  });
}

function switchView(viewName) {
  document.querySelectorAll(".nav-link").forEach(l => l.classList.remove("active"));
  document.querySelector(`[data-view="${viewName}"]`)?.classList.add("active");

  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  document.getElementById(`view-${viewName}`)?.classList.add("active");
}

// ================= HEALTH =================
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();

    const dot = document.querySelector(".status-dot");
    const text = document.getElementById("statusText");

    if (data.status === "ok") {
      dot.className = "status-dot online";
      text.textContent = `ML:${data.ml ? "✓" : "✗"} RAG:${data.rag ? "✓" : "✗"} LLM:${data.provider}`;
    }
  } catch {
    document.querySelector(".status-dot").className = "status-dot offline";
    document.getElementById("statusText").textContent = "Offline";
  }
}

// ================= ANALYZE =================
async function analyzeSymptoms() {
  const text = document.getElementById("symptomText").value.trim();

  if (!text) return showError("Enter symptoms first");
  if (text.length < 3) return showError("Add more detail");
  if (isAnalyzing) return;

  isAnalyzing = true;
  setLoadingState(true);
  simulateLoadingSteps();

  try {
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ text })
    });

    let data;
    try {
      data = await res.json();
    } catch {
      throw new Error("Invalid server response");
    }

    if (!res.ok || data.status !== "success") {
      throw new Error(data.message || "Analysis failed");
    }

    // ✅ CRITICAL FIX (REAL SESSION FROM BACKEND)
    currentSessionId = data.session_id;

    console.log("SESSION:", currentSessionId);

    renderResults(data);

  } catch (err) {
    console.error("Analyze Error:", err);
    showError(err.message);
  } finally {
    isAnalyzing = false;
    setLoadingState(false);
  }
}

// ================= LOADING =================
function setLoadingState(loading) {
  document.getElementById("loadingState").style.display = loading ? "flex" : "none";

  if (loading) {
    document.getElementById("emptyState").style.display = "none";
    document.getElementById("resultsContent").style.display = "none";
    document.getElementById("errorState").style.display = "none";
  }
}

function simulateLoadingSteps() {
  const steps = ["ls1","ls2","ls3","ls4"];
  steps.forEach((id,i) => {
    setTimeout(() => {
      document.getElementById(id)?.classList.add("active");
    }, i * 600);
  });
}

// ================= RESULTS =================
function renderResults(data) {
  document.getElementById("loadingState").style.display = "none";
  document.getElementById("emptyState").style.display = "none";
  document.getElementById("errorState").style.display = "none";
  document.getElementById("resultsContent").style.display = "block";

  renderSymptomTags(data.symptoms || []);
  renderPredictions(data.predictions || []);

  const llm = data.llm || {};

  document.getElementById("llmExplanation").textContent =
    llm.explanation || "No explanation available";

  // RED FLAGS
  const redFlags = llm.red_flags || [];
  const redFlagsEl = document.getElementById("redFlags");

  if (redFlags.length) {
    document.getElementById("redFlagsSection").style.display = "block";
    redFlagsEl.innerHTML = redFlags.map(f =>
      `<div class="red-flag-item">🚨 ${escapeHtml(f)}</div>`
    ).join("");
  } else {
    document.getElementById("redFlagsSection").style.display = "none";
  }

  // QUESTIONS
  document.getElementById("followUpQuestions").innerHTML =
    (llm.follow_up_questions || []).map((q,i)=>`
      <div class="followup-item">
        <div class="item-num">${i+1}</div>
        <div>${escapeHtml(q)}</div>
      </div>`).join("");

  // ACTIONS
  document.getElementById("recommendedActions").innerHTML =
    (llm.recommended_actions || []).map((a,i)=>`
      <div class="action-item">
        <div class="item-num">${i+1}</div>
        <div>${escapeHtml(a)}</div>
      </div>`).join("");

  document.getElementById("disclaimerText").textContent =
    llm.disclaimer || "";

  // enable chat
  document.getElementById("chatInput").disabled = false;
  document.getElementById("chatSendBtn").disabled = false;
}

// ================= SYMPTOMS =================
function renderSymptomTags(symptoms) {
  const el = document.getElementById("symptomTags");

  if (!symptoms.length) {
    el.innerHTML = `<span class="tag-placeholder">No symptoms detected</span>`;
    return;
  }

  el.innerHTML = symptoms.map(s =>
    `<span class="symptom-tag">${escapeHtml(s.replace(/_/g," "))}</span>`
  ).join("");
}

// ================= PREDICTIONS =================
function renderPredictions(predictions) {
  const el = document.getElementById("predictionsGrid");

  if (!predictions.length) {
    el.innerHTML = "No predictions found";
    return;
  }

  el.innerHTML = predictions.map((p,i)=>{
    const conf = p.confidence || 0;
    return `
      <div class="prediction-card ${i===0?'rank-1':''}">
        <div>
          <div class="pred-rank">#${i+1}</div>
          <div class="pred-disease">${escapeHtml(p.disease)}</div>
        </div>
        <div class="pred-confidence">
          <div class="conf-value">${conf.toFixed(1)}%</div>
          <div class="conf-bar">
            <div class="conf-bar-fill" style="width:${conf}%"></div>
          </div>
        </div>
      </div>`;
  }).join("");
}

// ================= QUICK ADD =================
function addQuickSymptom(symptom) {
  const textarea = document.getElementById("symptomText");
  textarea.value += (textarea.value ? ", " : "") + symptom;
}

// ================= CLEAR =================
function clearAll() {
  document.getElementById("symptomText").value = "";
  currentSessionId = null;

  document.getElementById("symptomTags").innerHTML =
    `<span class="tag-placeholder">Symptoms will appear here after analysis...</span>`;

  document.getElementById("resultsContent").style.display = "none";
  document.getElementById("emptyState").style.display = "flex";
  document.getElementById("errorState").style.display = "none";
}

// ================= CHAT =================
function startChat() {
  switchView("chat");
}

async function sendChat() {
  const input = document.getElementById("chatInput");
  const message = input.value.trim();

  if (!message) return;

  if (!currentSessionId) {
    appendMessage("assistant", "⚠ Please run analysis first.");
    return;
  }

  appendMessage("user", message);
  input.value = "";

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({
        message,
        session_id: currentSessionId
      })
    });

    const data = await res.json();

    appendMessage("assistant", data.response || "No response");

  } catch (err) {
    appendMessage("assistant", "Server error. Try again.");
  }
}

function appendMessage(role, text) {
  const container = document.getElementById("chatMessages");

  container.innerHTML += `
    <div class="chat-message ${role}">
      <div class="msg-avatar">${role==="user"?"🧑":"⚕"}</div>
      <div class="msg-bubble">${escapeHtml(text)}</div>
    </div>
  `;

  container.scrollTop = container.scrollHeight;
}

// ================= ERROR =================
function showError(msg) {
  document.getElementById("errorState").style.display = "flex";
  document.getElementById("errorMessage").textContent = msg;
  document.getElementById("loadingState").style.display = "none";
  document.getElementById("resultsContent").style.display = "none";
}

function clearError() {
  document.getElementById("errorState").style.display = "none";
}

// ================= UTIL =================
function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g,"&amp;")
    .replace(/</g,"&lt;")
    .replace(/>/g,"&gt;");
}