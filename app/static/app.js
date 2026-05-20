const state = {
  selectedFile: null,
  indexReady: false,
};

const allowedExtensions = new Set(["pdf", "txt", "md"]);

const els = {
  statusDot: document.querySelector("#statusDot"),
  statusLabel: document.querySelector("#statusLabel"),
  statusMeta: document.querySelector("#statusMeta"),
  globalAlert: document.querySelector("#globalAlert"),
  ingestForm: document.querySelector("#ingestForm"),
  dropZone: document.querySelector("#dropZone"),
  fileInput: document.querySelector("#fileInput"),
  fileTitle: document.querySelector("#fileTitle"),
  fileHint: document.querySelector("#fileHint"),
  recreateInput: document.querySelector("#recreateInput"),
  ingestButton: document.querySelector("#ingestButton"),
  ingestResult: document.querySelector("#ingestResult"),
  askStateLabel: document.querySelector("#askStateLabel"),
  queryForm: document.querySelector("#queryForm"),
  questionInput: document.querySelector("#questionInput"),
  queryButton: document.querySelector("#queryButton"),
  answerSection: document.querySelector("#answerSection"),
  answerText: document.querySelector("#answerText"),
  responseBadges: document.querySelector("#responseBadges"),
  sourcesSection: document.querySelector("#sourcesSection"),
  sourcesList: document.querySelector("#sourcesList"),
  traceSection: document.querySelector("#traceSection"),
  traceRequestId: document.querySelector("#traceRequestId"),
  traceSummary: document.querySelector("#traceSummary"),
  traceWarnings: document.querySelector("#traceWarnings"),
  traceChunks: document.querySelector("#traceChunks"),
  chips: document.querySelectorAll(".chip"),
};

function showAlert(message) {
  els.globalAlert.textContent = message;
  els.globalAlert.classList.remove("hidden");
}

function clearAlert() {
  els.globalAlert.textContent = "";
  els.globalAlert.classList.add("hidden");
}

function setLoading(button, isLoading, label) {
  button.disabled = isLoading;
  button.classList.toggle("is-loading", isLoading);
  const buttonLabel = button.querySelector(".button-label");
  if (buttonLabel && label) {
    buttonLabel.textContent = label;
  }
}

function setIndexReady(isReady, metaText = "") {
  state.indexReady = isReady;
  els.questionInput.disabled = !isReady;
  els.queryButton.disabled = !isReady;
  els.askStateLabel.textContent = isReady ? "Ready to query" : "Waiting for an index";
  els.questionInput.placeholder = isReady
    ? "Ask a grounded question about the ingested document..."
    : "Ingest a document first, then ask a question...";

  els.statusDot.classList.toggle("ready", isReady);
  els.statusDot.classList.toggle("error", false);
  els.statusLabel.textContent = isReady ? "Index ready" : "No indexed documents yet";
  els.statusMeta.textContent = metaText || (isReady ? "You can ask questions now." : "Upload and ingest a document to begin.");
}

function setStatusError(message) {
  els.statusDot.classList.remove("ready");
  els.statusDot.classList.add("error");
  els.statusLabel.textContent = "Status unavailable";
  els.statusMeta.textContent = message;
}

function fileExtension(fileName) {
  return fileName.split(".").pop().toLowerCase();
}

function selectFile(file) {
  if (!file) {
    return;
  }

  const ext = fileExtension(file.name);
  if (!allowedExtensions.has(ext)) {
    state.selectedFile = null;
    els.fileInput.value = "";
    els.fileTitle.textContent = "Choose a PDF, TXT, or MD file";
    els.fileHint.textContent = "This demo matches the document types currently supported by ingestion.";
    showAlert("Unsupported file type. Please upload a PDF, TXT, or MD file.");
    return;
  }

  clearAlert();
  state.selectedFile = file;
  els.fileTitle.textContent = file.name;
  els.fileHint.textContent = `${formatBytes(file.size)} selected`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes === 0) {
    return "0 bytes";
  }
  const units = ["bytes", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** index;
  return `${value.toFixed(value >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
}

async function parseResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return { detail: await response.text() };
}

function errorMessage(prefix, payload) {
  if (payload?.detail) {
    if (Array.isArray(payload.detail)) {
      return `${prefix}: ${payload.detail.map((item) => item.msg || JSON.stringify(item)).join(", ")}`;
    }
    return `${prefix}: ${payload.detail}`;
  }
  if (payload?.message) {
    return `${prefix}: ${payload.message}`;
  }
  return prefix;
}

async function refreshStatus() {
  try {
    const response = await fetch("/status/");
    const payload = await parseResponse(response);

    if (!response.ok) {
      throw new Error(errorMessage("Status check failed", payload));
    }

    if (payload.index_ready) {
      setIndexReady(true, `${payload.collection_name} has about ${payload.point_count} indexed chunks.`);
    } else if (payload.qdrant_error) {
      setIndexReady(false);
      setStatusError(payload.qdrant_error);
    } else {
      setIndexReady(false, `${payload.collection_name} is empty. Retrieval mode: ${payload.retrieval_mode}.`);
    }
  } catch (error) {
    setStatusError(error.message || "Could not reach the status endpoint.");
  }
}

async function handleIngest(event) {
  event.preventDefault();
  clearAlert();
  els.ingestResult.classList.add("hidden");

  if (!state.selectedFile) {
    showAlert("Choose a PDF, TXT, or MD file before ingesting.");
    return;
  }

  const formData = new FormData();
  formData.append("file", state.selectedFile);
  formData.append("recreate", els.recreateInput.checked ? "true" : "false");

  setLoading(els.ingestButton, true, "Ingesting...");
  els.queryButton.disabled = true;
  els.questionInput.disabled = true;

  try {
    const response = await fetch("/ingest/", {
      method: "POST",
      body: formData,
    });
    const payload = await parseResponse(response);

    if (!response.ok) {
      throw new Error(errorMessage("Ingestion failed", payload));
    }

    renderIngestResult(payload);
    els.ingestResult.classList.remove("hidden");
    setIndexReady(true, "Freshly ingested document is ready for questions.");
    await refreshStatus();
  } catch (error) {
    showAlert(error.message || "Ingestion failed. Check the backend logs for details.");
    await refreshStatus();
  } finally {
    setLoading(els.ingestButton, false, "Ingest document");
    els.queryButton.disabled = !state.indexReady;
    els.questionInput.disabled = !state.indexReady;
  }
}

async function handleQuery(event) {
  event.preventDefault();
  clearAlert();

  const query = els.questionInput.value.trim();
  if (!query) {
    showAlert("Enter a question before querying.");
    return;
  }

  setLoading(els.queryButton, true, "Thinking...");
  els.answerSection.classList.add("hidden");

  try {
    const response = await fetch("/query/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });
    const payload = await parseResponse(response);

    if (!response.ok) {
      throw new Error(errorMessage("Query failed", payload));
    }

    renderAnswer(payload);
  } catch (error) {
    showAlert(error.message || "Query failed. Check the backend logs for details.");
  } finally {
    setLoading(els.queryButton, false, "Ask question");
    els.queryButton.disabled = !state.indexReady;
  }
}

function renderAnswer(payload) {
  els.answerText.innerHTML = renderMarkdown(payload.answer || "No answer returned.");
  els.responseBadges.innerHTML = "";

  addBadge(payload.mode ? `Mode: ${payload.mode}` : "Mode: unknown");
  addBadge(payload.cached ? "Cached" : "Fresh");
  if (payload.score !== undefined && payload.score !== null) {
    addBadge(`Cache score: ${formatValue(payload.score)}`);
  }

  renderSources(collectSourceItems(payload));
  renderTrace(payload.trace);
  els.answerSection.classList.remove("hidden");
}

function renderMarkdown(markdown) {
  const lines = String(markdown).replace(/\r\n?/g, "\n").split("\n");
  const blocks = [];
  let paragraph = [];
  let listType = null;
  let listItems = [];
  let inCodeBlock = false;
  let codeLines = [];

  function flushParagraph() {
    if (paragraph.length > 0) {
      blocks.push(`<p>${renderInlineMarkdown(paragraph.join(" ").trim())}</p>`);
      paragraph = [];
    }
  }

  function flushList() {
    if (listItems.length > 0 && listType) {
      const items = listItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("");
      blocks.push(`<${listType}>${items}</${listType}>`);
      listItems = [];
      listType = null;
    }
  }

  lines.forEach((rawLine) => {
    const line = rawLine.trimEnd();
    const trimmed = line.trim();

    if (trimmed.startsWith("```")) {
      if (inCodeBlock) {
        blocks.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        codeLines = [];
        inCodeBlock = false;
      } else {
        flushParagraph();
        flushList();
        inCodeBlock = true;
      }
      return;
    }

    if (inCodeBlock) {
      codeLines.push(rawLine);
      return;
    }

    if (!trimmed) {
      flushParagraph();
      flushList();
      return;
    }

    const heading = trimmed.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const level = heading[1].length + 2;
      blocks.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      return;
    }

    const unorderedItem = trimmed.match(/^[-*]\s+(.+)$/);
    if (unorderedItem) {
      flushParagraph();
      if (listType && listType !== "ul") {
        flushList();
      }
      listType = "ul";
      listItems.push(unorderedItem[1].trim());
      return;
    }

    const orderedItem = trimmed.match(/^\d+[.)]\s+(.+)$/);
    if (orderedItem) {
      flushParagraph();
      if (listType && listType !== "ol") {
        flushList();
      }
      listType = "ol";
      listItems.push(orderedItem[1].trim());
      return;
    }

    flushList();
    paragraph.push(trimmed);
  });

  if (inCodeBlock) {
    blocks.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
  }
  flushParagraph();
  flushList();

  return blocks.join("");
}

function renderInlineMarkdown(text) {
  const codeSpans = [];
  let rendered = escapeHtml(text).replace(/`([^`]+)`/g, (_match, code) => {
    const token = `@@CODE_SPAN_${codeSpans.length}@@`;
    codeSpans.push(`<code>${code}</code>`);
    return token;
  });

  rendered = rendered
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/__(.+?)__/g, "<strong>$1</strong>")
    .replace(/(^|[^\*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/(^|[^_])_([^_\n]+)_/g, "$1<em>$2</em>");

  codeSpans.forEach((code, index) => {
    rendered = rendered.replace(`@@CODE_SPAN_${index}@@`, code);
  });

  return rendered;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function addBadge(text) {
  const badge = document.createElement("span");
  badge.className = "badge";
  badge.textContent = text;
  els.responseBadges.appendChild(badge);
}

function renderIngestResult(payload) {
  els.ingestResult.textContent = "";

  const title = document.createElement("strong");
  title.textContent = "Ingestion complete.";

  const summary = document.createElement("span");
  summary.textContent = `${payload.docs_ingested ?? 0} document page(s) loaded, ${payload.nodes ?? 0} chunks indexed.`;

  els.ingestResult.append(title, summary);

  if (payload.trace?.request_id) {
    const request = document.createElement("small");
    request.textContent = `Request ${payload.trace.request_id}`;
    els.ingestResult.appendChild(request);
  }
}

function collectSourceItems(payload) {
  const items = [];

  if (Array.isArray(payload.sources)) {
    payload.sources.forEach((source) => items.push(normalizeSource(source, "source")));
  }

  if (Array.isArray(payload.trace?.final_chunks)) {
    payload.trace.final_chunks.forEach((node) => items.push(normalizeSource(node, "final")));
  }

  if (Array.isArray(payload.reranked_nodes)) {
    payload.reranked_nodes.forEach((node) => items.push(normalizeSource(node, "reranked")));
  }

  if (items.length === 0 && Array.isArray(payload.retrieved_nodes)) {
    payload.retrieved_nodes.forEach((node) => items.push(normalizeSource(node, "retrieved")));
  }

  return items;
}

function renderTrace(trace) {
  if (!trace) {
    els.traceSection.classList.add("hidden");
    return;
  }

  els.traceRequestId.textContent = trace.request_id ? `Request ${trace.request_id}` : "";
  els.traceSummary.innerHTML = "";
  els.traceWarnings.innerHTML = "";
  els.traceChunks.innerHTML = "";

  const embeddingCalls = trace.external_calls?.embedding_calls || {};
  const llmCalls = trace.external_calls?.llm_calls || {};
  const rerankerCalls = trace.external_calls?.reranker_calls || {};

  const summaryItems = [
    ["LLM", `${trace.providers?.llm || "unknown"} / ${trace.models?.llm || "unknown"}`],
    ["Embeddings", `${trace.providers?.embedding || "unknown"} / ${trace.models?.embedding || "unknown"}`],
    ["Mode", trace.retrieval?.mode || trace.retrieval_mode || "unknown"],
    ["Top K", trace.retrieval?.top_k ?? "unknown"],
    ["Retrieved", trace.retrieved_count ?? "n/a"],
    ["Final Context", trace.final_context_count ?? "n/a"],
    ["Cache", trace.cache?.hit ? `hit (${formatValue(trace.cache.score)})` : trace.cache?.enabled ? "miss" : "off"],
    ["Latency", trace.timings?.total ? `${formatValue(trace.timings.total)}s` : "n/a"],
    ["Embedding Calls", Object.values(embeddingCalls).reduce((sum, value) => sum + Number(value || 0), 0)],
    ["LLM Calls", Object.values(llmCalls).reduce((sum, value) => sum + Number(value || 0), 0)],
    ["Remote Rerank Calls", rerankerCalls.remote ?? 0],
  ];

  summaryItems.forEach(([label, value]) => {
    els.traceSummary.appendChild(traceItem(label, value));
  });

  Object.entries(trace.timings || {}).forEach(([label, value]) => {
    els.traceSummary.appendChild(traceItem(`${label} latency`, `${formatValue(value)}s`));
  });

  if (Array.isArray(trace.warnings) && trace.warnings.length > 0) {
    trace.warnings.forEach((warning) => {
      const item = document.createElement("div");
      item.textContent = warning;
      els.traceWarnings.appendChild(item);
    });
    els.traceWarnings.classList.remove("hidden");
  } else {
    els.traceWarnings.classList.add("hidden");
  }

  const chunks = Array.isArray(trace.final_chunks) && trace.final_chunks.length
    ? trace.final_chunks
    : trace.retrieved_chunks || [];

  if (chunks.length > 0) {
    const title = document.createElement("h4");
    title.textContent = "Chunks used";
    els.traceChunks.appendChild(title);

    chunks.forEach((chunk) => {
      const row = document.createElement("div");
      row.className = "trace-chunk";
      const name = chunk.source || chunk.file_name || chunk.file_path || chunk.chunk_id || "chunk";
      const chunkTitle = document.createElement("strong");
      chunkTitle.textContent = String(name);
      const chunkMeta = document.createElement("span");
      chunkMeta.textContent = [
        chunk.chunk_id ? `id ${chunk.chunk_id}` : "",
        chunk.score !== undefined ? `score ${formatValue(chunk.score)}` : "",
        chunk.rerank_score !== undefined ? `rerank ${formatValue(chunk.rerank_score)}` : "",
        chunk.page_label || chunk.page_number || chunk.page ? `page ${chunk.page_label || chunk.page_number || chunk.page}` : "",
      ].filter(Boolean).join(" · ");
      row.append(chunkTitle, chunkMeta);
      els.traceChunks.appendChild(row);
    });
    els.traceChunks.classList.remove("hidden");
  } else {
    els.traceChunks.classList.add("hidden");
  }

  els.traceSection.classList.remove("hidden");
}

function traceItem(label, value) {
  const item = document.createElement("div");
  item.className = "trace-item";
  const key = document.createElement("span");
  key.textContent = label;
  const val = document.createElement("strong");
  val.textContent = value === undefined || value === null ? "n/a" : String(value);
  item.append(key, val);
  return item;
}

function normalizeSource(source, retrieverType) {
  if (!source || typeof source !== "object") {
    return { retriever_type: retrieverType, value: source };
  }

  const node = source.node && typeof source.node === "object" ? source.node : null;
  const metadata = node?.metadata && typeof node.metadata === "object" ? node.metadata : {};
  const normalized = {
    ...metadata,
    ...source,
    retriever_type: source.retriever_type || retrieverType,
  };

  if (node) {
    normalized.chunk_id = normalized.chunk_id || node.id_ || node.node_id || source.node_id;
    normalized.text = normalized.text || node.text;
  }

  return normalized;
}

function renderSources(sources) {
  els.sourcesList.innerHTML = "";

  if (!Array.isArray(sources) || sources.length === 0) {
    els.sourcesSection.classList.add("hidden");
    return;
  }

  sources.forEach((source, index) => {
    const card = document.createElement("article");
    card.className = "source-card";

    const title = document.createElement("div");
    title.className = "source-title";
    const sourceName = document.createElement("strong");
    sourceName.textContent = bestSourceName(source, index);
    const sourceNumber = document.createElement("span");
    sourceNumber.textContent = `Source ${index + 1}`;
    title.append(sourceName, sourceNumber);

    const grid = document.createElement("div");
    grid.className = "meta-grid";

    importantEntries(source).forEach(([key, value]) => {
      grid.appendChild(metaItem(key, value));
    });

    card.append(title);
    const chunkText = source.text || source.chunk || source.content;
    if (chunkText) {
      const snippet = document.createElement("p");
      snippet.className = "source-snippet";
      snippet.textContent = String(chunkText);
      card.appendChild(snippet);
    }
    card.appendChild(grid);
    els.sourcesList.appendChild(card);
  });

  els.sourcesSection.classList.remove("hidden");
}

function bestSourceName(source, index) {
  const candidates = [
    source.file_name,
    source.filename,
    source.file_path,
    source.path,
    source.document_id,
    source.doc_id,
    source.id,
  ];
  return candidates.find(Boolean) || `Retrieved chunk ${index + 1}`;
}

function importantEntries(source) {
  const preferred = [
    "file_name",
    "filename",
    "file_path",
    "path",
    "page_label",
    "page",
    "page_number",
    "chunk_id",
    "node_id",
    "document_id",
    "doc_id",
    "retriever",
    "retriever_type",
    "score",
    "similarity",
    "rerank_score",
    "sparse_score",
    "dense_score",
    "value",
  ];
  const entries = [];
  const seen = new Set();

  preferred.forEach((key) => {
    if (source[key] !== undefined && source[key] !== null && source[key] !== "") {
      entries.push([labelFor(key), source[key]]);
      seen.add(key);
    }
  });

  Object.entries(source).forEach(([key, value]) => {
    if (!seen.has(key) && key !== "node" && key !== "text" && key !== "chunk" && key !== "content" && value !== undefined && value !== null && value !== "") {
      entries.push([labelFor(key), value]);
    }
  });

  return entries.length ? entries : [["Metadata", "No source metadata returned"]];
}

function labelFor(key) {
  return key.replaceAll("_", " ");
}

function metaItem(key, value) {
  const item = document.createElement("div");
  item.className = "meta-item";
  const label = document.createElement("strong");
  label.textContent = key;
  const content = document.createElement("span");
  content.textContent = formatValue(value);
  item.append(label, content);
  return item;
}

function formatValue(value) {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

els.fileInput.addEventListener("change", (event) => {
  selectFile(event.target.files?.[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
  els.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropZone.classList.add("drag-over");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  els.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropZone.classList.remove("drag-over");
  });
});

els.dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files?.[0];
  if (file) {
    els.fileInput.files = event.dataTransfer.files;
    selectFile(file);
  }
});

els.ingestForm.addEventListener("submit", handleIngest);
els.queryForm.addEventListener("submit", handleQuery);

els.chips.forEach((chip) => {
  chip.addEventListener("click", () => {
    els.questionInput.value = chip.textContent;
    if (state.indexReady) {
      els.questionInput.focus();
    }
  });
});

refreshStatus();
