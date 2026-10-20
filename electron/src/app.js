/*
 * Renderer logic. One concern: talk to the FastAPI backend + show results.
 * All requests go to 127.0.0.1:8765 (same machine).
 */

const backendUrlPromise = window.pocketrag.getBackendUrl();

const $ = (id) => document.getElementById(id);

const STATUS_ENDPOINT = "/ingest_status/";
const POLL_MS = 1500;

$("btn-add-docs").addEventListener("click", addDocuments);
$("ask-form").addEventListener("submit", askQuestion);

// ---------- status bar / health ----------

async function checkBackend() {
  try {
    const base = await backendUrlPromise;
    const res = await fetch(`${base}/`);
    if (res.ok) {
      $("statusbar").textContent = "Backend ready";
    }
  } catch {
    $("statusbar").textContent = "Backend not reachable - is main.py running?";
  }
  refreshLibrary();
}

checkBackend();
setInterval(checkBackend, 5000);

// ---------- document library ----------

async function addDocuments() {
  const files = await window.pocketrag.pickFiles();
  if (!files || files.length === 0) return;

  const base = await backendUrlPromise;
  for (const filePath of files) {
    try {
      // file bytes come from the main process (renderer has no fs access)
      const buffer = await window.pocketrag.readFileBytes(filePath);
      const filename = filePath.split(/[\\/]/).pop();

      // multipart upload, same as a browser <form> post
      const form = new FormData();
      form.append(
        "file",
        new File([new Uint8Array(buffer)], filename),
        filename
      );

      const res = await fetch(`${base}/ingest/`, { method: "POST", body: form });
      const data = await res.json();
      console.log("ingest queued", filename, "->", data.doc_guid);
    } catch (e) {
      console.error("ingest failed:", e);
    }
  }
  refreshLibrary();
}

function makeStatus(text, cls) {
  return `<span class="status ${cls || ""}">${text}</span>`;
}

async function refreshLibrary() {
  try {
    const base = await backendUrlPromise;
    const res = await fetch(`${base}/documents`);
    const data = await res.json();

    const list = $("doc-list");
    list.innerHTML = "";
    if (!data.documents || data.documents.length === 0) {
      list.innerHTML = `<li class="empty">No documents yet</li>`;
      return;
    }
    for (const doc of data.documents) {
      const li = document.createElement("li");
      const ok = doc.status === "Complete";
      li.className = "doc";
      li.innerHTML = `
        <span class="name" title="${doc.filename}">${doc.filename}</span>
        ${makeStatus(doc.status, ok ? "ok" : "")}
        <button class="del" data-guid="${doc.doc_guid}">&times;</button>`;
      list.appendChild(li);
    }
    // delete buttons (event delegation on list)
    list.querySelectorAll(".del").forEach((btn) => {
      btn.addEventListener("click", () => deleteDocument(btn.dataset.guid));
    });
  } catch {
    // backend still starting - status bar already says so
  }
}

async function deleteDocument(guid) {
  const base = await backendUrlPromise;
  await fetch(`${base}/documents/${guid}`, { method: "DELETE" });
  refreshLibrary();
}

// ---------- chat / answers ----------

function addMessage(text, cls) {
  const div = document.createElement("div");
  div.className = cls;
  div.textContent = text;
  $("chat-log").appendChild(div);
  $("chat-log").scrollTop = $("chat-log").scrollHeight;
  return div;
}

function showSources(sources) {
  const panel = $("sources-panel");
  const list = $("sources-list");
  list.innerHTML = "";
  if (!sources || sources.length === 0) {
    panel.classList.add("hidden");
    return;
  }
  for (const source of sources) {
    const item = document.createElement("div");
    item.className = "source-item";
    item.innerHTML = `
      <div class="head">
        <span>[${source.number}] ${source.source} p${source.page}</span>
        <span>score ${source.rerank_score}</span>
      </div>
      <div class="snippet">${source.snippet}</div>`;
    list.appendChild(item);
  }
  panel.classList.remove("hidden");
}

async function askQuestion(event) {
  event.preventDefault();
  const question = $("question-input").value.trim();
  if (!question) return;

  addMessage(question, "user-msg");
  const typingBubble = addMessage("Thinking\u2026", "assistant-msg typing");
  $("btn-ask").disabled = true;

  try {
    const base = await backendUrlPromise;

    // endpoint: post query -> poll /answer/<guid>
    const res = await fetch(`${base}/query/?question=${encodeURIComponent(question)}`, { method: "POST" });
    const { query_guid } = await res.json();
    if (!query_guid) {
      typingBubble.textContent = res.status || "Backend busy";
      typingBubble.classList.remove("typing");
      $("btn-ask").disabled = false;
      return;
    }

    const answer = await pollForAnswer(query_guid);
    typingBubble.remove();
    if (answer) {
      addMessage(answer.answer, "assistant-msg");
      showSources(answer.sources);
    } else {
      addMessage("Ground away - timed out waiting for an answer.", "assistant-msg");
    }
  } catch (e) {
    typingBubble.remove();
    addMessage("Could not reach the backend: " + e, "assistant-msg");
  } finally {
    $("btn-ask").disabled = false;
    $("question-input").value = "";
  }
}

async function pollForAnswer(guid) {
  const base = await backendUrlPromise;
  for (let attempt = 0; attempt < 120; attempt++) {
    await sleep(POLL_MS);
    const res = await fetch(`${base}/answer/${guid}`);
    const { result } = await res.json();
    if (result) return result;
  }
  return null;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
