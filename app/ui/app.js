// Askwell demo UI. Vanilla JS, no build step. Every piece of model or corpus text is inserted
// with textContent / createTextNode only, never as markup.
"use strict";

const USER_KEY = "askwell.user";
const ALL_DEPARTMENTS = ["HR", "IT", "Operations", "Finance"];

const EXAMPLES = [
  { label: "Salary band G5 (restricted HR)", q: "What is the salary range for grade G5?" },
  { label: "Parental leave (public)", q: "How much parental leave do employees get?" },
  { label: "Security incident steps (confidential IT)", q: "What are the first steps in the security incident response runbook?" },
  { label: "Break-glass access (restricted IT)", q: "How does the privileged access break-glass procedure work?" },
  { label: "Travel policy (versions)", q: "What is the current travel and expense policy for meals?" },
  { label: "Guest Wi-Fi (public)", q: "How do visitors get on the guest Wi-Fi?" },
];

const $ = (id) => document.getElementById(id);
const els = {
  thread: $("thread"), input: $("input"), send: $("send"), stop: $("stop"),
  composer: $("composer"), newChat: $("new-chat"), select: $("user-select"),
  badge: $("user-badge"), panel: $("citation-panel"), panelTitle: $("panel-title"),
  panelBody: $("panel-body"), panelClose: $("panel-close"), examples: $("examples"),
  traceTemplate: $("trace-link-template"),
};

const state = { userId: "", me: null, sessionId: null, controller: null, busy: false };

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function storeGet() {
  try { return localStorage.getItem(USER_KEY) || ""; } catch (e) { return ""; }
}
function storeSet(value) {
  try { localStorage.setItem(USER_KEY, value); } catch (e) { /* private mode: ignore */ }
}

function headers(extra) {
  const h = Object.assign({}, extra || {});
  if (state.userId) h["X-User-Id"] = state.userId;
  return h;
}

// ---- errors ---------------------------------------------------------------------------

async function httpMessage(res) {
  let code = "", message = "";
  try {
    const body = await res.json();
    if (body && body.error) { code = body.error.code || ""; message = body.error.message || ""; }
  } catch (e) { /* not JSON */ }
  if (res.status === 401) {
    return "Unknown user (401). Pick a user from the list or switch to Guest.";
  }
  if (res.status === 403) return "Access denied (403): " + (message || "not allowed for this user.");
  if (res.status === 429) return "The model quota is exhausted (429). Try again later.";
  if (res.status === 503) return "Service unavailable: " + (message || code || "try again later.");
  return "Request failed (" + res.status + "): " + (message || code || res.statusText);
}

// ---- user switcher --------------------------------------------------------------------

function isAdmin(me) {
  return !!me && me.clearance === "restricted" &&
    ALL_DEPARTMENTS.every((d) => (me.departments || []).includes(d));
}

function renderBadge() {
  const me = state.me;
  if (!me) { els.badge.textContent = state.userId ? state.userId + " (unknown)" : "Guest"; return; }
  const depts = me.departments && me.departments.length ? me.departments.join(", ") : "none";
  els.badge.textContent =
    me.name + " | " + me.role + " | clearance: " + me.clearance + " | departments: " + depts;
}

async function loadMe() {
  state.me = null;
  try {
    const res = await fetch("/me", { headers: headers() });
    if (res.ok) state.me = await res.json();
    else addNotice(await httpMessage(res));
  } catch (e) {
    addNotice("Could not reach the server: " + e.message);
  }
  renderBadge();
}

async function loadUsers() {
  try {
    const res = await fetch("/users");
    if (!res.ok) throw new Error(await httpMessage(res));
    const users = await res.json();
    for (const u of users) {
      const opt = el("option", "", u.name + " (" + u.user_id + ", " + u.role + ")");
      opt.value = u.user_id;
      els.select.appendChild(opt);
    }
  } catch (e) {
    addNotice("Could not load users: " + e.message);
  }
  const saved = storeGet();
  const known = Array.from(els.select.options).some((o) => o.value === saved);
  state.userId = known ? saved : "";
  els.select.value = state.userId;
}

async function switchUser(userId) {
  if (state.controller) state.controller.abort();
  state.userId = userId;
  storeSet(userId);
  resetConversation();
  await loadMe();
}

// ---- conversation ---------------------------------------------------------------------

function resetConversation() {
  state.sessionId = null;
  els.thread.replaceChildren();
  closePanel();
}

function scrollDown() { els.thread.scrollTop = els.thread.scrollHeight; }

function addNotice(text) {
  const box = el("div", "msg error", text);
  els.thread.appendChild(box);
  scrollDown();
  return box;
}

function setBusy(busy) {
  state.busy = busy;
  els.input.disabled = busy;
  els.send.disabled = busy;
  els.stop.hidden = !busy;
  if (!busy) els.input.focus();
}

// Server-sent events from a fetch body: blocks are separated by a blank line.
async function* readEvents(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    buffer = buffer.replace(/\r\n/g, "\n");
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let event = "message";
      const data = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
      }
      if (data.length) {
        try { yield { event, data: JSON.parse(data.join("\n")) }; } catch (e) { /* skip bad block */ }
      }
    }
    if (done) return;
  }
}

async function ask(question) {
  question = question.trim();
  if (!question || state.busy) return;
  els.input.value = "";
  els.thread.appendChild(el("div", "msg user", question));
  const bubble = el("div", "msg");
  const text = el("div", "text");
  bubble.appendChild(text);
  els.thread.appendChild(bubble);
  scrollDown();
  setBusy(true);
  state.controller = new AbortController();
  let sources = [];
  let streamed = "";
  let finished = false;
  try {
    const res = await fetch("/chat/stream", {
      method: "POST",
      headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        question, include_text: true,
        session_id: state.sessionId || undefined,
      }),
      signal: state.controller.signal,
    });
    if (!res.ok) {
      showError(bubble, await httpMessage(res));
      return;
    }
    for await (const { event, data } of readEvents(res.body)) {
      if (event === "sources") {
        sources = data.sources || [];
      } else if (event === "token") {
        streamed += data.text || "";
        text.textContent = streamed;
        scrollDown();
      } else if (event === "done") {
        finished = true;
        if (data.session_id) state.sessionId = data.session_id;
        renderFinal(bubble, text, data, sources, question);
      } else if (event === "error") {
        finished = true;
        showError(bubble, "Error (" + (data.code || "unknown") + "): " + (data.message || ""));
      }
    }
    if (!finished) showError(bubble, "The stream ended before an answer was completed.");
  } catch (e) {
    if (e.name === "AbortError") {
      bubble.classList.add("stopped");
      if (!streamed) text.textContent = "(stopped)";
    } else {
      showError(bubble, "Network error: " + e.message);
    }
  } finally {
    state.controller = null;
    setBusy(false);
    scrollDown();
  }
}

function showError(bubble, message) {
  bubble.className = "msg error";
  bubble.replaceChildren(el("div", "text", message));
}

// ---- final answer ---------------------------------------------------------------------

// Source for marker n: sources are in rank order and the server numbers markers by rank
// (app/rag/answer.py parse_citations uses chunks[n - 1]).
function sourceFor(n, sources) {
  return sources.find((s) => s.rank === n) || sources[n - 1] || null;
}

function renderWithChips(container, answer, citations, sources) {
  const valid = new Set((citations || []).map((c) => c.number));
  container.replaceChildren();
  let last = 0;
  const marker = /\[(\d+)\]/g;
  let m;
  while ((m = marker.exec(answer)) !== null) {
    const n = Number(m[1]);
    const src = valid.has(n) ? sourceFor(n, sources) : null;
    if (!src) continue; // leave unknown markers as plain text
    container.appendChild(document.createTextNode(answer.slice(last, m.index)));
    const chip = el("button", "chip" + (src.status === "superseded" ? " old" : ""), "[" + n + "]");
    chip.type = "button";
    chip.title = src.title + (src.section ? " - " + src.section : "");
    chip.addEventListener("click", () => openPanel(n, src));
    container.appendChild(chip);
    last = m.index + m[0].length;
  }
  container.appendChild(document.createTextNode(answer.slice(last)));
}

function tag(text, className) { return el("span", "tag " + (className || ""), text); }

function renderFinal(bubble, text, data, sources, question) {
  if (data.refused) bubble.classList.add("refused");
  renderWithChips(text, data.answer || "", data.citations, sources);
  const meta = el("div", "meta");
  if (data.refused) {
    meta.appendChild(tag("refused" + (data.refusal_reason ? ": " + data.refusal_reason : ""), "refused"));
  }
  if (data.cached) meta.appendChild(tag("cached", "cached"));
  if (typeof data.latency_ms === "number") meta.appendChild(tag(data.latency_ms + " ms"));
  if (data.standalone_query && data.standalone_query !== question) {
    meta.appendChild(el("span", "rewrite", "searched as: " + data.standalone_query));
  }
  if (data.trace_id) meta.appendChild(traceLink(data.trace_id));
  bubble.appendChild(meta);
}

// ---- trace link -----------------------------------------------------------------------

function traceLink(traceId) {
  const link = els.traceTemplate.content.firstElementChild.cloneNode(true);
  link.textContent = "Trace " + traceId.slice(0, 8);
  if (!isAdmin(state.me)) {
    link.disabled = true;
    link.title = "Administrators only";
    return link;
  }
  link.title = "Open the trace view in a new tab";
  link.addEventListener("click", () => openTrace(traceId, link));
  return link;
}

// A plain <a href> cannot send X-User-Id, so fetch the HTML with the header and open a Blob URL.
async function openTrace(traceId, link) {
  const tab = window.open("", "_blank");
  link.disabled = true;
  try {
    const res = await fetch("/traces/" + encodeURIComponent(traceId) + "/view", { headers: headers() });
    if (!res.ok) throw new Error(await httpMessage(res));
    const blob = new Blob([await res.text()], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    if (tab) tab.location.href = url; else window.open(url, "_blank");
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  } catch (e) {
    if (tab) tab.close();
    addNotice("Could not open the trace: " + e.message);
  } finally {
    link.disabled = false;
  }
}

// ---- citation panel -------------------------------------------------------------------

function closePanel() { els.panel.hidden = true; els.panelBody.replaceChildren(); }

function openPanel(n, src) {
  els.panelTitle.textContent = "Source [" + n + "]";
  const body = els.panelBody;
  body.replaceChildren();
  body.appendChild(el("h3", "", src.title));
  if (src.status === "superseded") {
    body.appendChild(el("div", "banner", "Superseded: a newer version of this document exists."));
  }
  const dl = el("dl");
  const rows = [
    ["Section", src.section || "-"], ["Document", src.doc_id], ["Version", src.version || "-"],
    ["Status", src.status || "-"], ["Effective", src.effective_date || "-"],
    ["Score", typeof src.score === "number" ? src.score.toFixed(4) : "-"],
  ];
  for (const [k, v] of rows) {
    dl.appendChild(el("dt", "", k));
    const dd = el("dd", k === "Status" && v === "superseded" ? "status-superseded" : "", String(v));
    dl.appendChild(dd);
  }
  body.appendChild(dl);
  body.appendChild(el("div", "chunk", src.text || "(no chunk text returned)"));
  els.panel.hidden = false;
}

// ---- wiring ---------------------------------------------------------------------------

function renderExamples() {
  for (const ex of EXAMPLES) {
    const b = el("button", "", ex.label);
    b.type = "button";
    b.title = ex.q;
    b.addEventListener("click", () => ask(ex.q));
    els.examples.appendChild(b);
  }
}

els.composer.addEventListener("submit", (e) => { e.preventDefault(); ask(els.input.value); });
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); ask(els.input.value); }
});
els.stop.addEventListener("click", () => { if (state.controller) state.controller.abort(); });
els.newChat.addEventListener("click", () => {
  if (state.controller) state.controller.abort();
  resetConversation();
});
els.select.addEventListener("change", () => switchUser(els.select.value));
els.panelClose.addEventListener("click", closePanel);

(async function init() {
  renderExamples();
  await loadUsers();
  await loadMe();
  els.input.focus();
})();
