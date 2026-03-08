/* ─────────────────────────────────────────────────────────────
   Document Intelligence Refinery – Application Logic
   ───────────────────────────────────────────────────────────── */

const API = ""; // same-origin

// ── State ──────────────────────────────────────────────────────
let activeDocId = null;
let allFacts = [];

// ── Boot ───────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initNav();
  initQuery();
  initAudit();
  initFacts();
  initUpload();
  checkHealth();
  loadDocuments();
});

// ══════════════════════════════════════════════════════════════
// Navigation
// ══════════════════════════════════════════════════════════════

function initNav() {
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const panelId = btn.dataset.panel;
      switchPanel(panelId, btn);

      // Refresh facts when tab opens
      if (panelId === "panel-facts") loadFacts();
    });
  });
}

function switchPanel(panelId, btn) {
  document
    .querySelectorAll(".panel")
    .forEach((p) => p.classList.remove("active"));
  document
    .querySelectorAll(".nav-btn")
    .forEach((b) => b.classList.remove("active"));
  document.getElementById(panelId).classList.add("active");
  btn.classList.add("active");

  const titles = {
    "panel-query": "Query Interface",
    "panel-audit": "Audit Mode",
    "panel-facts": "Fact Table",
    "panel-upload": "Upload Document",
  };
  document.getElementById("page-title").textContent = titles[panelId] || "";
}

// ══════════════════════════════════════════════════════════════
// Health check
// ══════════════════════════════════════════════════════════════

async function checkHealth() {
  const el = document.getElementById("server-status");
  try {
    const r = await fetch(`${API}/api/health`);
    if (r.ok) {
      el.className = "status-dot status-ok";
      el.textContent = "Server online";
    } else {
      throw new Error();
    }
  } catch {
    el.className = "status-dot status-error";
    el.textContent = "Server offline";
  }
}

// ══════════════════════════════════════════════════════════════
// Documents list
// ══════════════════════════════════════════════════════════════

async function loadDocuments() {
  try {
    const r = await fetch(`${API}/api/documents`);
    const docs = await r.json();
    renderDocList(docs);
  } catch {
    /* ignore on fail */
  }
}

function renderDocList(docs) {
  const list = document.getElementById("doc-list");
  if (!docs.length) {
    list.innerHTML = '<div class="doc-empty">No documents yet</div>';
    return;
  }
  list.innerHTML = "";
  docs.forEach((d) => {
    const item = document.createElement("div");
    item.className = "doc-item" + (d.doc_id === activeDocId ? " active" : "");
    item.textContent = d.doc_id;
    item.title = `${d.ldu_count} LDUs`;
    item.addEventListener("click", () => selectDoc(d.doc_id, item));
    list.appendChild(item);
  });
}

function selectDoc(docId, el) {
  activeDocId = docId;
  document
    .querySelectorAll(".doc-item")
    .forEach((i) => i.classList.remove("active"));
  el.classList.add("active");
  const badge = document.getElementById("active-doc-badge");
  badge.classList.remove("hidden");
  document.getElementById("active-doc-name").textContent = docId;
}

// ══════════════════════════════════════════════════════════════
// Query Panel
// ══════════════════════════════════════════════════════════════

function initQuery() {
  document.getElementById("query-btn").addEventListener("click", runQuery);
  document.getElementById("query-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runQuery();
  });
  document
    .getElementById("provenance-toggle")
    .addEventListener("click", function () {
      toggleProvenance("provenance-cards", this);
    });
}

async function runQuery() {
  const question = document.getElementById("query-input").value.trim();
  if (!question) return;

  setLoading("query", true);
  hideEl("query-result");

  const body = { question };
  if (activeDocId) body.doc_id = activeDocId;

  try {
    const r = await fetch(`${API}/api/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const err = await r.json();
      throw new Error(err.detail || "Query failed");
    }
    const data = await r.json();
    renderQueryResult(data);
  } catch (err) {
    renderQueryResult({
      answer: `⚠ Error: ${err.message}`,
      provenance_chain: [],
      tool_trace: [],
    });
  } finally {
    setLoading("query", false);
  }
}

function renderQueryResult(data) {
  document.getElementById("answer-text").textContent =
    data.answer || "(no answer)";

  // Tool trace badges
  const tracesEl = document.getElementById("tool-trace-badges");
  tracesEl.innerHTML = "";
  (data.tool_trace || []).forEach((t) => {
    const chip = document.createElement("span");
    chip.className = "trace-chip";
    if (t.includes("navigate")) chip.classList.add("chip-navigate");
    if (t.includes("semantic")) chip.classList.add("chip-semantic");
    if (t.includes("structured")) chip.classList.add("chip-sql");
    chip.textContent = t;
    tracesEl.appendChild(chip);
  });

  // Provenance
  const pChain = data.provenance_chain || [];
  if (pChain.length) {
    document.getElementById("prov-count").textContent = pChain.length;
    renderProvenanceCards("provenance-cards", pChain);
    showEl("provenance-section");
  } else {
    hideEl("provenance-section");
  }

  showEl("query-result");
}

// ══════════════════════════════════════════════════════════════
// Audit Panel
// ══════════════════════════════════════════════════════════════

function initAudit() {
  document.getElementById("audit-btn").addEventListener("click", runAudit);
  document.getElementById("audit-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runAudit();
  });
  document
    .getElementById("audit-prov-toggle")
    .addEventListener("click", function () {
      toggleProvenance("audit-provenance-cards", this);
    });
}

async function runAudit() {
  const claim = document.getElementById("audit-input").value.trim();
  if (!claim) return;

  setLoading("audit", true);
  hideEl("audit-result");

  const body = { claim };
  if (activeDocId) body.doc_id = activeDocId;

  try {
    const r = await fetch(`${API}/api/audit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const err = await r.json();
      throw new Error(err.detail || "Audit failed");
    }
    const data = await r.json();
    renderAuditResult(data);
  } catch (err) {
    renderAuditResult({
      verified: false,
      answer: `⚠ Error: ${err.message}`,
      provenance_chain: [],
      audit_score: 0,
    });
  } finally {
    setLoading("audit", false);
  }
}

function renderAuditResult(data) {
  const verified = data.verified;
  const score = data.audit_score ?? 0;

  // Verdict badge
  const badge = document.getElementById("audit-verdict-badge");
  if (verified === true) {
    badge.className = "audit-badge verified";
    badge.innerHTML = "✓ VERIFIED";
  } else if (verified === false) {
    badge.className = "audit-badge unverifiable";
    badge.innerHTML = "⚠ UNVERIFIABLE";
  } else {
    badge.className = "audit-badge";
    badge.innerHTML = "— UNKNOWN";
  }

  // Score bar
  const pct = Math.min(100, Math.round(score * 100));
  document.getElementById("audit-score-fill").style.width = `${pct}%`;
  document.getElementById("audit-score-label").textContent =
    `Similarity: ${score.toFixed(3)}`;

  document.getElementById("audit-answer").textContent = data.answer || "";

  // Provenance
  const pChain = data.provenance_chain || [];
  if (pChain.length) {
    document.getElementById("audit-prov-count").textContent = pChain.length;
    renderProvenanceCards("audit-provenance-cards", pChain);
    showEl("audit-provenance");
  } else {
    hideEl("audit-provenance");
  }

  showEl("audit-result");
}

// ══════════════════════════════════════════════════════════════
// Fact Table
// ══════════════════════════════════════════════════════════════

function initFacts() {
  document.getElementById("facts-refresh").addEventListener("click", loadFacts);
  document.getElementById("fact-search").addEventListener("input", function () {
    filterFacts(this.value.toLowerCase());
  });
}

async function loadFacts() {
  setLoading("facts", true);
  const params = activeDocId
    ? `?doc_id=${encodeURIComponent(activeDocId)}`
    : "";
  try {
    const r = await fetch(`${API}/api/facts${params}`);
    if (!r.ok) throw new Error("Failed to load facts");
    allFacts = await r.json();
    renderFacts(allFacts);
  } catch {
    allFacts = [];
    renderFacts([]);
  } finally {
    setLoading("facts", false);
  }
}

function filterFacts(query) {
  if (!query) {
    renderFacts(allFacts);
    return;
  }
  const filtered = allFacts.filter(
    (f) =>
      f.key.toLowerCase().includes(query) ||
      (f.value || "").toLowerCase().includes(query) ||
      (f.doc_id || "").toLowerCase().includes(query),
  );
  renderFacts(filtered);
}

function renderFacts(facts) {
  const tbody = document.getElementById("facts-tbody");
  if (!facts.length) {
    tbody.innerHTML =
      '<tr><td colspan="6" class="empty-row">No facts extracted yet. Upload a document first.</td></tr>';
    return;
  }
  tbody.innerHTML = facts
    .map(
      (f) => `
    <tr>
      <td>${esc(f.key)}</td>
      <td>${esc(f.value)}</td>
      <td>${esc(f.unit || "—")}</td>
      <td>${esc(f.period || "—")}</td>
      <td>p.${esc(String(f.page_no))}</td>
      <td title="${esc(f.doc_id)}">${esc(truncate(f.doc_id, 28))}</td>
    </tr>
  `,
    )
    .join("");
}

// ══════════════════════════════════════════════════════════════
// Upload Panel
// ══════════════════════════════════════════════════════════════

function initUpload() {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const uploadCard = document.getElementById("upload-card");

  // Click to browse
  uploadCard.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) uploadFile(fileInput.files[0]);
  });

  // Drag-and-drop
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () =>
    dropZone.classList.remove("dragover"),
  );
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file && file.name.toLowerCase().endsWith(".pdf")) uploadFile(file);
  });
}

const STEPS = ["step-triage", "step-extract", "step-chunk", "step-facts"];
const STEP_LINES = ["line-1", "line-2", "line-3"];

function setStep(idx) {
  STEPS.forEach((s, i) => {
    const el = document.getElementById(s);
    if (i < idx) {
      el.classList.add("done");
      el.classList.remove("active");
    } else if (i === idx) {
      el.classList.add("active");
      el.classList.remove("done");
    } else {
      el.classList.remove("done", "active");
    }
  });
  // Animate step-lines too
  document.querySelectorAll(".step-line").forEach((l, i) => {
    l.classList.toggle("done", i < idx);
  });
}

async function uploadFile(file) {
  showEl("upload-progress");
  hideEl("upload-result");
  hideEl("upload-card");
  document.getElementById("upload-status-text").textContent =
    `Uploading ${file.name}…`;
  setStep(0);

  const formData = new FormData();
  formData.append("file", file);

  // Fake step animation while waiting for the server
  const statusMsgs = [
    "Running triage analysis…",
    "Extracting document content…",
    "Chunking into LDUs…",
    "Extracting numerical facts…",
  ];
  let stepIdx = 0;
  const stepTimer = setInterval(() => {
    if (stepIdx < 4) {
      setStep(stepIdx);
      document.getElementById("upload-status-text").textContent =
        statusMsgs[stepIdx];
      stepIdx++;
    }
  }, 2000);

  try {
    const r = await fetch(`${API}/api/upload`, {
      method: "POST",
      body: formData,
    });
    clearInterval(stepTimer);
    setStep(4); // all done

    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: "Unknown error" }));
      throw new Error(err.detail || "Upload failed");
    }
    const data = await r.json();
    renderUploadResult(data);
  } catch (err) {
    clearInterval(stepTimer);
    document.getElementById("upload-status-text").textContent =
      `⚠ ${err.message}`;
    STEPS.forEach((s) =>
      document.getElementById(s).classList.remove("done", "active"),
    );
  }
}

function renderUploadResult(data) {
  hideEl("upload-progress");
  showEl("upload-card");

  const el = document.getElementById("upload-result");
  el.innerHTML = `
    <h3>✓ Document processed successfully</h3>
    <dl class="result-kv">
      <dt>Document ID</dt>   <dd>${esc(data.doc_id)}</dd>
      <dt>Strategy</dt>      <dd>${esc(data.strategy)}</dd>
      <dt>Escalated</dt>     <dd>${data.escalated ? "Yes" : "No"}</dd>
      <dt>LDUs</dt>          <dd>${esc(String(data.ldu_count))}</dd>
      <dt>Facts extracted</dt><dd>${esc(String(data.fact_count))}</dd>
      <dt>Domain</dt>        <dd>${esc(data.profile?.domain_hint || "—")}</dd>
      <dt>Language</dt>      <dd>${esc(data.profile?.primary_language || "—")}</dd>
      <dt>Pages</dt>         <dd>${esc(String(data.profile?.total_pages || "—"))}</dd>
    </dl>
  `;
  showEl("upload-result");
  loadDocuments();
}

// ══════════════════════════════════════════════════════════════
// Shared provenance renderer
// ══════════════════════════════════════════════════════════════

function renderProvenanceCards(containerId, chain) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  chain.forEach((c, i) => {
    const card = document.createElement("div");
    card.className = "provenance-card";
    card.style.animationDelay = `${i * 0.06}s`;

    const toolName = c.tool_used || "unknown";
    const toolClass = toolName.includes("navigate")
      ? "chip-navigate"
      : toolName.includes("semantic")
        ? "chip-semantic"
        : toolName.includes("query")
          ? "chip-sql"
          : "";

    let bboxStr = "";
    if (c.bbox) {
      const b = c.bbox;
      bboxStr = `<span class="prov-tag">bbox: (${round(b.x0)}, ${round(b.y0)}) → (${round(b.x1)}, ${round(b.y1)})</span>`;
    }
    let hashStr = "";
    if (c.content_hash) {
      hashStr = `<div class="prov-hash">hash: ${c.content_hash.slice(0, 32)}…</div>`;
    }

    card.innerHTML = `
      <div class="prov-meta">
        <span class="prov-tag prov-tool-tag ${toolClass}">⚙ ${esc(toolName)}</span>
      </div>
      <div class="prov-meta">
        <span class="prov-tag prov-page-tag">p.${esc(String(c.page_number))}</span>
        ${bboxStr}
      </div>
      <div class="prov-meta">
        <span class="prov-tag">${esc(c.document_name || "—")}</span>
      </div>
      ${c.snippet ? `<div class="prov-snippet">${esc(c.snippet)}</div>` : ""}
      ${hashStr}
    `;
    container.appendChild(card);
  });
}

function toggleProvenance(cardsId, toggleBtn) {
  const cards = document.getElementById(cardsId);
  const isOpen = cards.classList.toggle("open");
  toggleBtn.classList.toggle("open", isOpen);
}

// ══════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════

function setLoading(panel, on) {
  const loadId = `${panel}-loading`;
  const btnId = `${panel}-btn`;
  if (on) {
    showEl(loadId);
    if (document.getElementById(btnId))
      document.getElementById(btnId).disabled = true;
  } else {
    hideEl(loadId);
    if (document.getElementById(btnId))
      document.getElementById(btnId).disabled = false;
  }
}

function showEl(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove("hidden");
}

function hideEl(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add("hidden");
}

function esc(str) {
  return (str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function truncate(str, n) {
  return str && str.length > n ? str.slice(0, n) + "…" : str;
}

function round(n) {
  return Math.round(n * 10) / 10;
}
