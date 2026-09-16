/**
 * AskTube popup.
 *
 * Flow:
 *   1. read the active tab's URL          (chrome.tabs API)
 *   2. check the FastAPI server is alive  (GET /health)
 *   3. send question + URL to the server  (POST /ask)
 *   4. render the answer
 *
 * Manifest V3 forbids inline scripts and inline onclick handlers, which is why
 * every listener is attached here with addEventListener instead.
 */

const API = "http://localhost:8000";

const els = {
  videoLine: document.getElementById("video-line"),
  serverDot: document.getElementById("server-dot"),
  question: document.getElementById("question"),
  ask: document.getElementById("ask"),
  output: document.getElementById("output"),
  answer: document.getElementById("answer"),
  meta: document.getElementById("meta"),
  error: document.getElementById("error"),
};

let currentUrl = null;

/** Any YouTube URL shape -> the 11-character video id, or null. */
function extractVideoId(url) {
  const patterns = [
    /[?&]v=([A-Za-z0-9_-]{11})/,
    /youtu\.be\/([A-Za-z0-9_-]{11})/,
    /youtube\.com\/(?:embed|shorts|live)\/([A-Za-z0-9_-]{11})/,
  ];
  for (const pattern of patterns) {
    const match = url.match(pattern);
    if (match) return match[1];
  }
  return null;
}

function showError(message) {
  els.error.textContent = message;
  els.error.hidden = false;
  els.output.hidden = true;
}

function clearError() {
  els.error.hidden = true;
  els.error.textContent = "";
}

/** Is the backend running? Colours the dot in the header. */
async function checkServer() {
  try {
    const response = await fetch(`${API}/health`);
    if (!response.ok) throw new Error(String(response.status));
    const data = await response.json();
    els.serverDot.className = "dot dot-ok";
    els.serverDot.title = `server ok - ${data.videos_cached} video(s) cached`;
    return true;
  } catch {
    els.serverDot.className = "dot dot-bad";
    els.serverDot.title = "server not reachable";
    showError(
      "Backend not running.\n\nStart it with:\n  cd backend\n  uvicorn api:app --reload --port 8000"
    );
    return false;
  }
}

/** Ask the backend. Throws with a readable message on failure. */
async function ask(url, question) {
  const response = await fetch(`${API}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, question }),
  });

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail[0]?.msg ?? detail;
    } catch {
      /* response had no JSON body - keep the status code */
    }
    throw new Error(detail);
  }

  return response.json();
}

async function onAsk() {
  const question = els.question.value.trim();
  if (!question || !currentUrl) return;

  clearError();
  els.ask.disabled = true;
  els.ask.textContent = "Thinking…";
  els.output.hidden = true;

  try {
    const result = await ask(currentUrl, question);
    els.answer.textContent = result.answer;
    els.meta.textContent = `${result.video_id} · ${result.seconds}s`;
    els.output.hidden = false;
  } catch (err) {
    showError(err.message);
  } finally {
    els.ask.disabled = false;
    els.ask.textContent = "Ask";
  }
}

/** Runs when the popup opens. */
async function init() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const videoId = tab?.url ? extractVideoId(tab.url) : null;

  if (!videoId) {
    els.videoLine.textContent = "Open a YouTube video first";
    return;
  }

  currentUrl = tab.url;
  els.videoLine.textContent = `video ${videoId}`;

  if (await checkServer()) {
    els.question.disabled = false;
    els.ask.disabled = false;
    els.question.focus();
  }
}

els.ask.addEventListener("click", onAsk);

// Cmd/Ctrl + Enter submits, so you never have to reach for the mouse.
els.question.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") onAsk();
});

init();
