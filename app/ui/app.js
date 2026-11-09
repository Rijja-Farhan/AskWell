// Askwell demo UI. Vanilla JS, no build step. Every piece of model or corpus text is inserted
// with textContent / createTextNode only, never as markup.
"use strict";

const USER_KEY = "askwell.user";
const THEME_KEY = "askwell.theme";
const ALL_DEPARTMENTS = ["HR", "IT", "Operations", "Finance"];

const EXAMPLES = [
  { level: "restricted", q: "What is the salary range for grade G5?" },
  { level: "public", q: "How much parental leave do employees get?" },
  { level: "confidential", q: "What are the first steps in the security incident response runbook?" },
  { level: "restricted", q: "How does the privileged access break-glass procedure work?" },
  { level: "versions", q: "What is the current travel and expense policy for meals?" },
  { level: "public", q: "How do visitors get on the guest Wi-Fi?" },
];

const $ = (id) => document.getElementById(id);
const els = {
  scroll: $("scroll"), welcome: $("welcome"), thread: $("thread"), input: $("input"),
  send: $("send"), stop: $("stop"), composer: $("composer"), newChat: $("new-chat"),
  select: $("user-select"), badge: $("user-badge"), avatar: $("user-avatar"),
  panel: $("citation-panel"), panelTitle: $("panel-title"), panelBody: $("panel-body"),
  panelClose: $("panel-close"), examples: $("examples"), themeToggle: $("theme-toggle"),
  traceTemplate: $("trace-link-template"), markTemplate: $("mark-template"),
};

const state = { userId: "", me: null, sessionId: null, controller: null, busy: false };

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function storeGet(key) {
  try { return localStorage.getItem(key) || ""; } catch (e) { return ""; }
}
function storeSet(key, value) {
  try { localStorage.setItem(key, value); } catch (e) { /* private mode: ignore */ }
}

function headers(extra) {
  const h = Object.assign({}, extra || {});
  if (state.userId) h["X-User-Id"] = state.userId;
  return h;
}

// ---- theme ----------------------------------------------------------------------------

// system -> light -> dark -> system. "system" removes the attribute so the OS setting decides.
function applyTheme(theme) {
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
  const label = "Theme: " + (theme || "system");
  els.themeToggle.title = label;
  els.themeToggle.setAttribute("aria-label", label);
}
function cycleTheme() {
  const current = document.documentElement.dataset.theme || "";
  const next = current === "" ? "light" : current === "light" ? "dark" : "";
  storeSet(THEME_KEY, next);
  applyTheme(next);
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

function initials(name) {
  const parts = String(name || "G").trim().split(/\s+/).filter(Boolean);
  return ((parts[0] || "G")[0] + (parts.length > 1 ? parts[parts.length - 1][0] : "")).toUpperCase();
}

function renderBadge() {
  const me = state.me;
  els.badge.replaceChildren();
  if (!me) {
    els.avatar.textContent = state.userId ? "?" : "G";
    els.badge.appendChild(el("span", "pill", state.userId ? state.userId + " (unknown)" : "Guest"));
    return;
  }
  els.avatar.textContent = initials(me.name);
  els.badge.appendChild(el("span", "", me.role));
  els.badge.appendChild(el("span", "pill level", me.clearance));
  const depts = me.departments && me.departments.length ? me.departments : ["no departments"];
  for (const d of depts) els.badge.appendChild(el("span", "pill", d));
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
  const saved = storeGet(USER_KEY);
  const known = Array.from(els.select.options).some((o) => o.value === saved);
  state.userId = known ? saved : "";
  els.select.value = state.userId;
}

async function switchUser(userId) {
  if (state.controller) state.controller.abort();
  state.userId = userId;
  storeSet(USER_KEY, userId);
  resetConversation();
  await loadMe();
}

// ---- conversation ---------------------------------------------------------------------

function resetConversation() {
  state.sessionId = null;
  els.thread.replaceChildren();
  els.welcome.hidden = false;
  closePanel();
}

function scrollDown() { els.scroll.scrollTop = els.scroll.scrollHeight; }

function addNotice(text) {
  const row = el("div", "msg error");
  row.appendChild(el("div", "bubble", text));
  els.thread.appendChild(row);
  scrollDown();
  return row;
}

function setBusy(busy) {
  state.busy = busy;
  els.input.disabled = busy;
  els.send.disabled = busy;
  els.send.hidden = busy;
  els.stop.hidden = !busy;
  if (!busy) els.input.focus();
}

function growInput() {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, 160) + "px";
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

// One assistant message: [mark] [bubble: status | text, sources, meta].
function newBotMessage() {
  const row = el("div", "msg bot streaming");
  row.appendChild(els.markTemplate.content.firstElementChild.cloneNode(true));
  const bubble = el("div", "bubble");
  const status = el("div", "status");
  const dots = el("span", "dots");
  for (let i = 0; i < 3; i++) dots.appendChild(el("i"));
  const statusText = el("span", "", "Searching documents...");
  status.append(dots, statusText);
  const text = el("div", "text");
  text.hidden = true;
  bubble.append(status, text);
  row.appendChild(bubble);
  return { row, bubble, status, statusText, text };
}

async function ask(question) {
  question = question.trim();
  if (!question || state.busy) return;
  els.input.value = "";
  growInput();
  els.welcome.hidden = true;
  const userRow = el("div", "msg user");
  userRow.appendChild(el("div", "bubble", question));
  els.thread.appendChild(userRow);
  const msg = newBotMessage();
  els.thread.appendChild(msg.row);
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
      showError(msg, await httpMessage(res));
      return;
    }
    for await (const { event, data } of readEvents(res.body)) {
      if (event === "sources") {
        sources = data.sources || [];
        msg.statusText.textContent = sources.length
          ? "Reading " + sources.length + (sources.length === 1 ? " source..." : " sources...")
          : "No matching documents found...";
      } else if (event === "token") {
        if (!streamed) { msg.status.hidden = true; msg.text.hidden = false; }
        streamed += data.text || "";
        msg.text.textContent = streamed;
        scrollDown();
      } else if (event === "done") {
        finished = true;
        if (data.session_id) state.sessionId = data.session_id;
        renderFinal(msg, data, sources, question);
      } else if (event === "error") {
        finished = true;
        showError(msg, "Error (" + (data.code || "unknown") + "): " + (data.message || ""));
      }
    }
    if (!finished) showError(msg, "The stream ended before an answer was completed.");
  } catch (e) {
    if (e.name === "AbortError") {
      msg.row.classList.add("stopped");
      msg.status.hidden = true;
      msg.text.hidden = false;
      if (!streamed) msg.text.textContent = "(stopped)";
    } else {
      showError(msg, "Network error: " + e.message);
    }
  } finally {
    msg.row.classList.remove("streaming");
    state.controller = null;
    setBusy(false);
    scrollDown();
  }
}

function showError(msg, message) {
  msg.row.className = "msg error";
  msg.row.replaceChildren(el("div", "bubble", message));
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
    // Drop the space before the marker so the number chip sits against the word it supports.
    container.appendChild(document.createTextNode(answer.slice(last, m.index).replace(/[ \t]+$/, "")));
    const chip = el("button", "cite" + (src.status === "superseded" ? " old" : ""), String(n));
    chip.type = "button";
    chip.title = src.title + (src.section ? " - " + src.section : "");
    chip.setAttribute("aria-label", "Source " + n + ": " + src.title);
    chip.addEventListener("click", () => openPanel(n, src));
    container.appendChild(chip);
    last = m.index + m[0].length;
  }
  container.appendChild(document.createTextNode(answer.slice(last)));
}

// Cards for the sources the answer actually cites, in citation order.
function renderSources(bubble, citations, sources) {
  const numbers = Array.from(new Set((citations || []).map((c) => c.number))).sort((a, b) => a - b);
  const cards = [];
  for (const n of numbers) {
    const src = sourceFor(n, sources);
    if (!src) continue;
    const old = src.status === "superseded";
    const card = el("button", "source" + (old ? " old" : ""));
    card.type = "button";
    card.appendChild(el("span", "n", String(n)));
    card.appendChild(el("span", "t", src.title));
    card.appendChild(el("span", "s",
      (src.section ? src.section + " | " : "") + "v" + (src.version || "?") + (old ? " (superseded)" : "")));
    card.addEventListener("click", () => openPanel(n, src));
    cards.push(card);
  }
  if (!cards.length) return;
  const wrap = el("div", "sources");
  wrap.appendChild(el("div", "sources-label", cards.length === 1 ? "Source" : "Sources"));
  const list = el("div", "source-list");
  list.append(...cards);
  wrap.appendChild(list);
  bubble.appendChild(wrap);
}

function tag(text, className) { return el("span", "tag " + (className || ""), text); }

function renderFinal(msg, data, sources, question) {
  msg.status.hidden = true;
  msg.text.hidden = false;
  if (data.refused) msg.row.classList.add("refused");
  renderWithChips(msg.text, data.answer || "", data.citations, sources);
  if (!data.refused) renderSources(msg.bubble, data.citations, sources);
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
  msg.bubble.appendChild(meta);
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
  els.panelTitle.textContent = "Source " + n;
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
    const b = el("button", "example");
    b.type = "button";
    b.appendChild(el("span", "q", ex.q));
    const lvl = el("span", "lvl", ex.level === "versions" ? "multiple versions" : ex.level);
    lvl.dataset.l = ex.level;
    b.appendChild(lvl);
    b.addEventListener("click", () => ask(ex.q));
    els.examples.appendChild(b);
  }
}

els.composer.addEventListener("submit", (e) => { e.preventDefault(); ask(els.input.value); });
els.input.addEventListener("input", growInput);
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
els.themeToggle.addEventListener("click", cycleTheme);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !els.panel.hidden) closePanel(); });

(async function init() {
  applyTheme(storeGet(THEME_KEY));
  renderExamples();
  await loadUsers();
  await loadMe();
  els.input.focus();
})();
