/* Manga Translator — front-end logic */

const $ = (id) => document.getElementById(id);
const state = { files: [], jobId: null, poll: null };

/* ---------------- meta ---------------- */
async function loadMeta() {
  const meta = await (await fetch("/api/meta")).json();

  const src = $("sourceLang"), tgt = $("targetLang");
  meta.source_languages.forEach((l) => src.add(new Option(l.name, l.code)));
  meta.target_languages.forEach((l) => tgt.add(new Option(l.name, l.code)));
  tgt.value = "en";

  const dev = meta.gpu_available ? meta.device : "cpu";
  $("deviceBadge").innerHTML = `device <b>${dev}</b>` +
    (meta.llm_configured ? ` · llm <b>${meta.llm_model}</b>` : " · google fallback");

  const packs = meta.font_packs;
  $("fontPacks").innerHTML = Object.keys(packs).length
    ? Object.entries(packs)
        .map(([k, v]) => `<div><code>${k}/</code> ${v.join(", ")}</div>`)
        .join("") +
      `<div style="margin-top:8px">Drop .ttf/.otf files into <code>fonts/&lt;lang&gt;/</code> to add your own packs.</div>`
    : `No packs found. Drop .ttf/.otf files into <code>fonts/&lt;lang&gt;/</code> — e.g. <code>fonts/en/</code>.`;
}

/* ---------------- file picking ---------------- */
const pickers = { files: $("pickFiles"), folder: $("pickFolder"), zip: $("pickZip") };
document.querySelectorAll("[data-pick]").forEach((b) =>
  b.addEventListener("click", (e) => {
    e.stopPropagation();
    pickers[b.dataset.pick].click();
  })
);
Object.values(pickers).forEach((inp) =>
  inp.addEventListener("change", () => addFiles([...inp.files]))
);

const dz = $("dropzone");
dz.addEventListener("click", () => pickers.files.click());
["dragenter", "dragover"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); })
);
["dragleave", "drop"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("dragover"); })
);
dz.addEventListener("drop", (e) => addFiles([...e.dataTransfer.files]));

const isImage = (f) => /\.(png|jpe?g|webp|bmp|tiff?)$/i.test(f.name);
const isZip = (f) => /\.zip$/i.test(f.name);

function addFiles(files) {
  const ok = files.filter((f) => isImage(f) || isZip(f));
  if (files.length && !ok.length) return toast("Only images or .zip archives");
  if (ok.some(isZip) && ok.length > 1) return toast("A zip must be uploaded on its own");
  state.files = ok;
  state.jobId = null;
  $("results").classList.remove("on");
  $("queue").classList.add("on");
  $("queueCount").textContent = `${ok.length} file${ok.length > 1 ? "s" : ""}`;
  $("queueCurrent").textContent = ok.length
    ? ok.slice(0, 3).map((f) => f.webkitRelativePath || f.name).join(", ") +
      (ok.length > 3 ? ` … +${ok.length - 3} more` : "")
    : "";
  setProgress(0);
}

/* ---------------- job lifecycle ---------------- */
$("translateBtn").addEventListener("click", startJob);

async function startJob() {
  if (!state.files.length) return toast("Add some pages first");
  const fd = new FormData();
  state.files.forEach((f) =>
    // keep folder structure: webkitRelativePath travels as the filename
    fd.append("files", f, f.webkitRelativePath || f.name)
  );
  fd.append("source_lang", $("sourceLang").value);
  fd.append("target_lang", $("targetLang").value);
  fd.append("translator", $("translator").value);
  fd.append("direction", $("direction").value);
  fd.append("uppercase", $("uppercase").checked);
  fd.append("device", $("device").value);
  fd.append("font_path", $("fontPath").value);
  fd.append("llm_api_key", $("llmApiKey").value);
  fd.append("llm_base_url", $("llmBaseUrl").value);
  fd.append("llm_model", $("llmModel").value);

  $("translateBtn").disabled = true;
  $("queueCurrent").textContent = "uploading…";
  try {
    const res = await fetch("/api/jobs", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    state.jobId = (await res.json()).job_id;
    state.poll = setInterval(pollJob, 1000);
  } catch (e) {
    toast("Upload failed: " + e.message);
    $("translateBtn").disabled = false;
  }
}

async function pollJob() {
  let job;
  try {
    const res = await fetch(`/api/jobs/${state.jobId}`);
    if (!res.ok) throw new Error(await res.text());
    job = await res.json();
  } catch (e) {
    $("queueCurrent").textContent = "status check failed: " + e.message;
    return;                                   // keep polling — don't die silently
  }
  $("queueCount").textContent = job.total ? `${job.done} / ${job.total}` : "…";

  const stage = job.stage || job.status;
  $("queueCurrent").textContent = stage;
  $("stageLabel").textContent = stage ? `· ${stage}` : "";

  // smooth progress: page completion + fraction through the current page's stages
  if (job.total) {
    const frac = Math.min(1, Math.max(0, job.stage_frac || 0));
    setProgress(((job.done + (job.done < job.total ? frac * 0.99 : 0)) / job.total) * 100);
  } else if (job.status === "running") {
    setProgress(5);                           // moving, but total not known yet
  }

  if (job.status === "done" || job.status === "error") {
    clearInterval(state.poll);
    state.poll = null;
    $("translateBtn").disabled = false;
    $("stageLabel").textContent = "";
    setProgress(job.status === "done" ? 100 : 0);
    if (job.status === "error") return toast(job.error || "Job failed");
    renderResults(job);
  }
}

function setProgress(pct) { $("progressBar").style.width = pct + "%"; }

/* ---------------- results ---------------- */
function renderResults(job) {
  const grid = $("resultsGrid");
  grid.innerHTML = "";
  $("results").classList.add("on");

  const ok = job.results.filter((r) => !r.error);
  const dl = $("downloadBtn");
  if (ok.length) {
    dl.style.display = "";
    dl.onclick = () => (location.href = `/api/jobs/${job.id}/download`);
  }

  job.results.forEach((r, i) => {
    const tile = document.createElement("div");
    tile.className = "tile" + (r.error ? " error" : "");
    tile.style.animationDelay = i * 0.06 + "s";

    if (r.error) {
      tile.innerHTML = `<div class="tile-meta"><span class="tile-name">${r.src}</span><span>failed</span></div>
        <div class="tile-texts on"><div class="pair"><div class="src">${r.error}</div></div></div>`;
    } else {
      const pairs = (r.texts || [])
        .map((t) => `<div class="pair"><div class="src">${esc(t.src)}</div><div class="tgt">${esc(t.tgt)}</div></div>`)
        .join("");
      const untranslated = (r.texts || []).filter((t) => t.src && t.tgt === t.src).length;
      tile.innerHTML = `
        <img src="${r.url}" alt="${r.src}" loading="lazy" />
        <div class="tile-meta">
          <span class="tile-name" title="${r.out}">${r.out}</span>
          <span>${r.translated ?? r.regions}/${r.regions} translated${untranslated ? ` · <b style="color:#e0a940">${untranslated} kept original</b>` : ""}</span>
        </div>
        ${pairs ? `<div class="tile-texts">${pairs}</div>` : "<div class=\"tile-texts on\"><div class=\"pair\"><div class=\"src\">No text detected on this page</div></div></div>"}`;
      tile.querySelector("img").addEventListener("click", () => openLightbox(r.url));
      if (pairs) {
        const meta = tile.querySelector(".tile-meta");
        meta.style.cursor = "pointer";
        meta.title = "toggle source/translation text";
        meta.addEventListener("click", () =>
          tile.querySelector(".tile-texts").classList.toggle("on"));
      }
    }
    grid.appendChild(tile);
  });
  toast(`Done — ${ok.length} page${ok.length === 1 ? "" : "s"} translated`);
}

const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ---------------- lightbox & toast ---------------- */
function openLightbox(url) {
  $("lightboxImg").src = url;
  $("lightbox").classList.add("on");
}
$("lightbox").addEventListener("click", () => $("lightbox").classList.remove("on"));

let toastTimer;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("on");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("on"), 3200);
}

loadMeta();
