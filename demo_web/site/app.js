/* Argus · Where on Earth? — replays REAL precomputed pipeline outputs
   (demo_web/export_demo_data.py). Nothing here invents results: every
   similarity, inlier count, correspondence, footprint and error comes from
   data/q/<id>.json. The only decorative element is the descriptor bar
   animation during the embed step. */

const AUTHOR = {
  name: "Priyanka Vijaybaskar",
  links: [
    // { label: "LinkedIn", href: "https://www.linkedin.com/in/..." },
    // { label: "GitHub", href: "https://github.com/..." },
  ],
};

const TEX = "https://cdn.jsdelivr.net/npm/three-globe/example/img/";
// Reference tiles are rebuilt from EOx's public Sentinel-2 cloudless WMTS (same imagery the reference DB was built from).
const EOX = (year, z, y, x) => `https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-${year}_3857/default/g/${z}/${y}/${x}.jpg`;
const EOX_ATTR = 'Sentinel-2 cloudless 2021 by <a href="https://s2maps.eu" target="_blank" rel="noopener">EOX IT Services GmbH</a> (contains modified Copernicus Sentinel data 2021)';
const ISS_KMS = 7.66;
const EARTH_R = 6371;
const DEG = Math.PI / 180;
const SNAP_WARN_KM = 900;
const PACE = 2.4; // replay slow-down vs. real GPU wall-clock, so humans can follow it

const S = {
  photos: [], stats: null, current: null, detail: null,
  mode: "explore", guess: null, running: false, skip: false, ran: false,
  selectedCand: null, score: loadScore(),
};

const $ = (s) => document.querySelector(s);
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const fmt = (n, d = 0) => Number(n).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const sleep = (ms) => (S.skip ? Promise.resolve() : new Promise((r) => setTimeout(r, ms)));

function haversine(a, b, c, d) {
  const dLat = (c - a) * DEG, dLon = (d - b) * DEG;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a * DEG) * Math.cos(c * DEG) * Math.sin(dLon / 2) ** 2;
  return 2 * EARTH_R * Math.asin(Math.sqrt(h));
}
function fmtLatLon(lat, lon) {
  return `${Math.abs(lat).toFixed(4)}° ${lat >= 0 ? "N" : "S"}, ${Math.abs(lon).toFixed(4)}° ${lon >= 0 ? "E" : "W"}`;
}
function fmtKm(km) {
  if (km < 1) return `${fmt(km * 1000)} m`;
  if (km < 100) return `${fmt(km, 1)} km`;
  return `${fmt(km)} km`;
}
function tween(dur, fn) {
  return new Promise((res) => {
    if (S.skip || dur <= 0) { fn(1); return res(); }
    const t0 = performance.now();
    const step = (t) => {
      const k = Math.min(1, (t - t0) / dur);
      fn(1 - Math.pow(1 - k, 3));
      if (k < 1 && !S.skip) requestAnimationFrame(step); else { fn(1); res(); }
    };
    requestAnimationFrame(step);
  });
}
async function typeText(node, text, cps = 60) {
  node.textContent = "";
  for (let i = 0; i < text.length; i++) {
    if (S.skip) { node.textContent = text; return; }
    node.textContent = text.slice(0, i + 1);
    await new Promise((r) => setTimeout(r, 1000 / cps));
  }
}
function toast(msg, ms = 3800) {
  const t = $("#toast");
  t.innerHTML = msg;
  t.classList.add("on");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("on"), ms);
}
function loadScore() {
  try { return JSON.parse(localStorage.getItem("argus-score")) || { rounds: 0, argus: 0, you: 0 }; }
  catch { return { rounds: 0, argus: 0, you: 0 }; }
}
function saveScore() { try { localStorage.setItem("argus-score", JSON.stringify(S.score)); } catch {} }

/* ------------------------------------------------------------------ boot */
async function boot() {
  const log = $("#boot-log"), fill = $("#boot-fill");
  const line = async (html, pct) => { log.innerHTML += html + "\n"; fill.style.width = pct + "%"; await new Promise((r) => setTimeout(r, 260)); };

  await line('<span class="hl">argus-loc</span> :: vision-based localization from orbit', 8);
  const manifest = await fetch("data/manifest.json").then((r) => r.json());
  S.photos = manifest.photos;
  S.stats = manifest.stats;
  const tiles = Object.values(S.stats.db_tiles || {}).reduce((a, b) => a + b, 0);
  await line(`> reference map ....... <span class="ok">${Object.keys(S.stats.regions).length} regions · ${fmt(tiles)} satellite tiles</span>`, 25);
  await line('> retriever ........... <span class="ok">DINOv2-B/14 + SALAD (LoRA fine-tuned)</span>', 40);
  await line('> matcher ............. <span class="ok">SIFT + LightGlue · MAGSAC</span>', 52);
  await line(`> photos .............. <span class="ok">${S.photos.length} astronaut shots from the ISS</span>`, 64);
  await line('> GPS ................. <span class="hl">disabled (that\'s the whole point)</span>', 76);
  await line("> rendering Earth ...", 84);

  initGlobe(() => {
    fill.style.width = "100%";
    log.innerHTML += '<span class="ok">> ready.</span>\n';
    setTimeout(() => $("#boot").classList.add("done"), 450);
  });
  initTicker();
  initHood();
  initCredit();
  renderScore();
  setState("idle");
}

/* ------------------------------------------------------------------ globe */
let globe, sat = { lat: 0, lng: 0, alt: 0.066, kind: "sat" }, guessPin = null;
const paths = { zones: [], orbit: null, footprints: [] };

function circlePts(lat, lon, rKm, n = 128, alt = 0.004) {
  const d = rKm / EARTH_R, p1 = lat * DEG, l1 = lon * DEG, pts = [];
  for (let i = 0; i <= n; i++) {
    const th = (2 * Math.PI * i) / n;
    const p2 = Math.asin(Math.sin(p1) * Math.cos(d) + Math.cos(p1) * Math.sin(d) * Math.cos(th));
    const l2 = l1 + Math.atan2(Math.sin(th) * Math.sin(d) * Math.cos(p1), Math.cos(d) - Math.sin(p1) * Math.sin(p2));
    pts.push([p2 / DEG, ((l2 / DEG + 540) % 360) - 180, alt]);
  }
  return pts;
}
const INC = 51.6 * DEG, NODE = -30;
function orbitAt(u) {
  const lat = Math.asin(Math.sin(INC) * Math.sin(u)) / DEG;
  const lng = NODE + Math.atan2(Math.cos(INC) * Math.sin(u), Math.cos(u)) / DEG;
  return [lat, ((lng + 540) % 360) - 180];
}

function refreshPaths() {
  globe.pathsData([...paths.zones, paths.orbit, ...paths.footprints].filter(Boolean));
}

function initGlobe(onReady) {
  const regions = S.stats.regions, R = S.stats.region_radius_km || 2500;
  paths.zones = Object.entries(regions).map(([name, [la, lo]]) => ({
    pts: circlePts(la, lo, R), color: ["rgba(94,242,255,0.05)", "rgba(94,242,255,0.75)"], stroke: 0.7, dash: 0.02, gap: 0.012, anim: 16000,
  }));
  const orbitPts = [];
  for (let i = 0; i <= 256; i++) { const [a, b] = orbitAt((2 * Math.PI * i) / 256); orbitPts.push([a, b, sat.alt]); }
  paths.orbit = { pts: orbitPts, color: "rgba(177,140,255,0.55)", stroke: 0.35, dash: 0.006, gap: 0.006, anim: 60000 };

  const zoneLabels = Object.entries(regions).map(([name, [la, lo]]) => ({ lat: la, lng: lo, text: name.toUpperCase(), kind: "zone" }));
  S.zoneLabels = zoneLabels;

  globe = Globe({ animateIn: true })(document.getElementById("globe"))
    .globeImageUrl(TEX + "earth-blue-marble.jpg")
    .bumpImageUrl(TEX + "earth-topology.png")
    .backgroundImageUrl(TEX + "night-sky.png")
    .atmosphereColor("#5ef2ff").atmosphereAltitude(0.2)
    .pointsData(S.photos).pointLat("lat").pointLng("lon")
    .pointAltitude((d) => (S.current && d.id === S.current.id && S.mode !== "mystery" ? 0.05 : 0.008))
    .pointRadius((d) => (S.current && d.id === S.current.id && S.mode !== "mystery" ? 0.35 : 0.22))
    .pointColor((d) => (S.current && d.id === S.current.id && S.mode !== "mystery" ? "#ffffff" : "rgba(94,242,255,0.85)"))
    .pointsTransitionDuration(400)
    .pointLabel((d) => (S.mode === "mystery" ? "" : `<div style="font:12px JetBrains Mono,monospace;background:rgba(0,0,0,.7);padding:4px 8px;border-radius:6px">ISS photo · ${d.region}</div>`))
    .onPointClick((d) => onGlobeClick(d.lat, d.lon))
    .onGlobeClick(({ lat, lng }) => onGlobeClick(lat, lng))
    .pathPoints("pts").pathPointLat((p) => p[0]).pathPointLng((p) => p[1]).pathPointAlt((p) => p[2])
    .pathColor("color").pathStroke("stroke").pathDashLength("dash").pathDashGap("gap").pathDashAnimateTime("anim")
    .pathTransitionDuration(0)
    .labelsData(zoneLabels).labelLat("lat").labelLng("lng").labelText("text")
    .labelSize((d) => (d.kind === "zone" ? 1.1 : 0.9)).labelDotRadius((d) => (d.kind === "zone" ? 0.25 : 0.45))
    .labelColor((d) => (d.kind === "zone" ? "rgba(94,242,255,0.75)" : d.color)).labelResolution(3).labelAltitude(0.01)
    .ringLat("lat").ringLng("lng").ringColor((d) => (t) => d.color.replace("A", String(1 - t)))
    .ringMaxRadius("r").ringPropagationSpeed("speed").ringRepeatPeriod("period")
    .arcStartLat("a0").arcStartLng("b0").arcEndLat("a1").arcEndLng("b1")
    .arcColor(() => ["rgba(177,140,255,0.9)", "rgba(255,179,71,0.9)"]).arcStroke(0.5)
    .arcDashLength(0.35).arcDashGap(0.15).arcDashAnimateTime(1600).arcAltitudeAutoScale(0.35)
    .htmlElementsData([sat]).htmlLat("lat").htmlLng("lng").htmlAltitude("alt").htmlElement(makeHtml)
    .onGlobeReady(onReady);

  refreshPaths();
  globe.pointOfView({ lat: 25, lng: 10, altitude: 2.4 });
  const ctl = globe.controls();
  ctl.autoRotate = true; ctl.autoRotateSpeed = 0.35; ctl.enableDamping = true;
  ctl.addEventListener("start", () => { ctl.autoRotate = false; $("#globe-hint").classList.add("off"); });

  const resize = () => { const g = document.getElementById("globe"); globe.width(g.clientWidth).height(g.clientHeight); };
  window.addEventListener("resize", resize); resize();

  let u = 0.4;
  setInterval(() => {
    u += 0.004;
    [sat.lat, sat.lng] = orbitAt(u);
    globe.htmlElementsData(guessPin ? [sat, guessPin] : [sat]);
  }, 50);
}

function makeHtml(d) {
  if (d.kind === "pin") {
    const e = el("div", "pin", "📍");
    return e;
  }
  return el("div", "sat",
    `<svg viewBox="0 0 32 32"><g fill="none" stroke="#5ef2ff" stroke-width="1.6"><rect x="12" y="12" width="8" height="8" fill="rgba(94,242,255,.3)"/><path d="M12 16H3M20 16h9M5 12v8M27 12v8M16 12V8"/></g><circle cx="16" cy="7" r="1.6" fill="#5ef2ff"/></svg><span>ARGUS</span>`);
}

function setRings(list) { globe.ringsData(list); }
function setLabels(extra = []) { globe.labelsData([...S.zoneLabels, ...extra]); }
function flyTo(lat, lng, altitude, ms = 1600) { globe.controls().autoRotate = false; globe.pointOfView({ lat, lng, altitude }, ms); }
function fpPath(corners, color, stroke = 0.6, dashed = false) {
  const pts = [...corners, corners[0]].map(([a, b]) => [a, b, 0.006]);
  return { pts, color, stroke, dash: dashed ? 0.04 : 1, gap: dashed ? 0.03 : 0, anim: dashed ? 2500 : 0 };
}

/* ------------------------------------------------------------------ states */
function setState(name) {
  document.querySelectorAll(".state").forEach((s) => s.classList.toggle("on", s.dataset.state === name));
  $("#panel").scrollTop = 0;
}

function nearestPhoto(lat, lon) {
  let best = null, bd = Infinity;
  for (const p of S.photos) {
    const d = haversine(lat, lon, p.lat, p.lon);
    if (d < bd) { bd = d; best = p; }
  }
  return [best, bd];
}

function onGlobeClick(lat, lng) {
  $("#globe-hint").classList.add("off");
  if (S.running) return;
  if (S.mode === "mystery" && S.current && !S.ran) {
    S.guess = { lat, lng };
    guessPin = { lat, lng, alt: 0.01, kind: "pin" };
    $("#guess-status").innerHTML = `Guess locked at <b>${fmtLatLon(lat, lng)}</b>. Click again to move it, or hit Localize.`;
    $("#btn-localize").disabled = false;
    return;
  }
  const [p, d] = nearestPhoto(lat, lng);
  if (!p) return;
  if (d > SNAP_WARN_KM) {
    toast(`Argus doesn't have a reference map there yet. Snapping to the nearest photo, <b>${fmt(d)} km</b> away in the <b>${p.region}</b> zone.`);
  }
  selectPhoto(p, "explore");
}

function resetRun() {
  S.running = false; S.skip = false; S.ran = false; S.detail = null; S.selectedCand = null;
  $("#pipeline").classList.add("hidden");
  $("#result").classList.add("hidden");
  $("#btn-localize").classList.remove("hidden", "busy");
  $("#btn-skip").classList.remove("hidden");
  $("#btn-localize").disabled = false;
  $("#cands").innerHTML = "";
  $("#descriptor").innerHTML = "";
  $("#search-counter").classList.add("hidden");
  document.querySelectorAll("#stepper li").forEach((li) => li.classList.remove("active", "done"));
  $("#photo-wrap").classList.remove("scanning");
  $("#patches").classList.remove("on");
  paths.footprints = []; refreshPaths();
  globe.arcsData([]); setRings([]); setLabels();
}

async function selectPhoto(p, mode) {
  S.mode = mode;
  S.current = p;
  S.guess = null; guessPin = null;
  resetRun();
  setState("photo");
  globe.pointsData([...S.photos]);

  const mystery = mode === "mystery";
  $("#kicker").textContent = mystery ? "BEAT THE AI" : "STEP 2 · LOCALIZE";
  $("#mystery-banner").classList.toggle("hidden", !mystery);
  $("#guess-status").textContent = "Look at the photo, then click the globe where you think it was taken.";
  $("#btn-localize").disabled = mystery;

  const img = $("#q-img");
  img.classList.remove("loaded");
  img.removeAttribute("src");
  $("#photo-tag").textContent = "fetching from NASA…";

  if (!mystery) {
    flyTo(p.lat, p.lon, 0.85);
    setRings([{ lat: p.lat, lng: p.lon, color: "rgba(94,242,255,A)", r: 4, speed: 2.5, period: 900 }]);
  } else {
    flyTo(20, (Math.random() * 300) - 150, 2.3, 1200);
  }

  const detail = await fetch(`data/q/${p.id}.json`).then((r) => r.json());
  if (S.current !== p) return;
  S.detail = detail;
  loadPhoto(detail);
  renderMeta(detail, mystery);
  if (!mystery) {
    globe.arcsData([{ a0: detail.nadir[0], b0: detail.nadir[1], a1: detail.truth_center[0], b1: detail.truth_center[1] }]);
    setLabels([{ lat: detail.nadir[0], lng: detail.nadir[1], text: "ISS overhead", color: "rgba(177,140,255,0.95)", kind: "iss" }]);
  }
}

function renderMeta(d, mystery) {
  const width = Math.sqrt(d.area_km2);
  const hide = (v) => (mystery ? `<b class="blur">${v}</b>` : `<b>${v}</b>`);
  $("#q-meta").innerHTML = `
    <div><span>Taken by</span><b>${d.mission}</b></div>
    <div><span>Date</span><b>${d.date}</b></div>
    <div><span>Region</span>${hide(d.region)}</div>
    <div><span>Covers about</span><b>${fmt(width)} × ${fmt(width)} km</b></div>`;
  $("#photo-tag").innerHTML = mystery
    ? "NASA photo · location hidden 🤫"
    : `<a href="${d.gape_page}" target="_blank" rel="noopener">${d.photo_id} · NASA GAPE ↗</a>`;
}

function loadPhoto(d) {
  const img = $("#q-img");
  img.onload = () => {
    S.aspect = img.naturalWidth / img.naturalHeight;
    $("#photo-wrap").style.aspectRatio = String(S.aspect);
    img.classList.add("loaded");
  };
  img.onerror = () => {
    $("#photo-tag").innerHTML = `NASA's photo server didn't answer. <a href="${d.gape_page}" target="_blank" rel="noopener">open on GAPE ↗</a>`;
  };
  img.src = d.gape_img;
}

/* A reference tile is a 4x4 block of web-mercator tiles at zoom z starting at (x, y).
   `drop` renders it from coarser zoom (z - drop) so small thumbnails fetch 1-4 images, not 16. */
function tileMosaic(c, drop) {
  const { z, y, x, year } = c.wmts;
  const dz = Math.min(drop, z), zz = z - dz, sc = 2 ** dz, span = 4 / sc;
  const x0 = x / sc, y0 = y / sc;
  const box = el("div", "mosaic");
  for (let ty = Math.floor(y0); ty < Math.ceil(y0 + span - 1e-9); ty++) {
    for (let tx = Math.floor(x0); tx < Math.ceil(x0 + span - 1e-9); tx++) {
      const im = el("img");
      im.loading = "lazy"; im.alt = "";
      im.src = EOX(year, zz, ty, tx);
      Object.assign(im.style, {
        left: `${((tx - x0) / span) * 100}%`, top: `${((ty - y0) / span) * 100}%`,
        width: `${100 / span}%`, height: `${100 / span}%`,
      });
      box.appendChild(im);
    }
  }
  return box;
}

/* ------------------------------------------------------------------ pipeline replay */
function setStep(name) {
  const order = ["embed", "search", "match", "geo"];
  const i = order.indexOf(name);
  document.querySelectorAll("#stepper li").forEach((li, j) => {
    li.classList.toggle("done", j < i || name === "done");
    li.classList.toggle("active", j === i);
  });
}
const status = (html) => { $("#status-text").innerHTML = html; };

async function localize() {
  if (S.running || !S.detail) return;
  const d = S.detail;
  S.running = true; S.skip = false;
  $("#btn-localize").classList.add("busy");
  await sleep(350);
  $("#btn-localize").classList.add("hidden");
  $("#pipeline").classList.remove("hidden");
  $("#pipeline").scrollIntoView({ behavior: "smooth", block: "start" });

  /* 01 embed */
  setStep("embed");
  status("Embedding photo → 2048-d descriptor…");
  const wrap = $("#photo-wrap"), patches = $("#patches");
  wrap.classList.add("scanning");
  patches.innerHTML = "";
  const cells = Array.from({ length: 256 }, () => patches.appendChild(el("i")));
  patches.classList.add("on");
  const desc = $("#descriptor");
  desc.innerHTML = "";
  const bars = Array.from({ length: 64 }, () => { const b = el("i"); b.style.height = "2px"; return desc.appendChild(b); });
  const embedMs = Math.max(1300, d.timings.embed_ms * PACE * 3);
  const t0 = performance.now();
  while (!S.skip && performance.now() - t0 < embedMs) {
    for (let k = 0; k < 18; k++) cells[(Math.random() * 256) | 0].classList.toggle("lit");
    bars.forEach((b) => (b.style.height = 3 + Math.random() * 23 + "px"));
    await sleep(70);
  }
  cells.forEach((c) => c.classList.remove("lit"));
  patches.classList.remove("on");
  wrap.classList.remove("scanning");
  status(`Descriptor ready <span class="muted">(${fmt(d.timings.embed_ms)} ms on GPU)</span>`);
  await sleep(350);

  /* 02 search */
  setStep("search");
  const tiles = (S.stats.db_tiles || {})[d.region] || 0;
  const cmp = tiles * 4;
  const counter = $("#search-counter");
  counter.classList.remove("hidden");
  status(`Searching the ${d.region} reference map…`);
  await tween(1200, (k) => { counter.innerHTML = `<b>${fmt(k * cmp)}</b> tile views compared <span class="muted">(${fmt(tiles)} tiles × 4 rotations)</span>`; });
  const sims = d.candidates.map((c) => c.sim), smin = Math.min(...sims), smax = Math.max(...sims);
  const grid = $("#cands");
  grid.innerHTML = "";
  const cards = d.candidates.map((c, i) => {
    const card = el("div", "cand", `<span class="rk">#${i + 1}</span><span class="sim"><i style="height:${20 + 80 * (c.sim - smin) / Math.max(1e-6, smax - smin)}%"></i></span><span class="inl">0</span>`);
    card.prepend(tileMosaic(c, 2));
    card.title = `rank ${i + 1} · cosine similarity ${c.sim.toFixed(3)}`;
    card.addEventListener("click", () => S.ran && showCandidate(i));
    grid.appendChild(card);
    return card;
  });
  for (const c of cards) { c.classList.add("in"); await sleep(55); }
  status(`Top-15 shortlist <span class="muted">(search ${fmt(d.timings.search_ms)} ms)</span>. Now: do the pixels actually line up?`);
  await sleep(700);

  /* 03 match */
  setStep("match");
  let leader = -1;
  for (let i = 0; i < d.candidates.length; i++) {
    const c = d.candidates[i], card = cards[i];
    card.classList.add("matching");
    const inl = card.querySelector(".inl");
    card.classList.add("scored");
    status(`Matching #${i + 1}/15 · ${c.matches} keypoint matches → RANSAC…`);
    const dur = Math.min(900, Math.max(260, c.match_ms * PACE));
    await tween(dur, (k) => { inl.textContent = `${Math.round(k * c.inliers)} inl`; });
    card.classList.remove("matching");
    card.classList.add(c.inliers >= d.min_inliers ? "pass" : "fail");
    if (leader < 0 || c.inliers > d.candidates[leader].inliers) {
      if (leader >= 0) cards[leader].classList.remove("best");
      leader = i;
      card.classList.add("best");
    }
    if (c.inliers >= d.min_inliers) card.classList.remove("fail");
  }
  const best = d.candidates[d.best];
  cards.forEach((c, i) => c.classList.toggle("best", i === d.best));
  status(`Best: #${d.best + 1} with <b>${best.inliers}</b> inliers <span class="muted">(needs ≥ ${d.min_inliers})</span>`);
  await sleep(700);

  /* 04 georeference */
  setStep("geo");
  status(d.status === "fix" ? "Homography → pixel-to-ground mapping → footprint…" : "Not enough geometric agreement. Refusing to guess.");
  await sleep(800);
  setStep("done");
  status(d.status === "fix" ? "Done ✓" : "Done: no fix");
  $("#btn-skip").classList.add("hidden");

  S.running = false; S.ran = true;
  await showResult();
}

/* ------------------------------------------------------------------ result */
async function showResult() {
  const d = S.detail, fix = d.status === "fix", mystery = S.mode === "mystery";
  const res = $("#result");
  res.classList.remove("hidden");
  const v = $("#verdict");
  const best = d.candidates[d.best];

  const guessKm = mystery && S.guess ? haversine(S.guess.lat, S.guess.lng, d.truth_center[0], d.truth_center[1]) : null;

  if (fix) {
    v.className = "verdict fix";
    v.innerHTML = `FIX ACQUIRED<small>${best.inliers} matched features agree on where this is.</small>`;
  } else {
    v.className = "verdict nofix";
    v.innerHTML = `NO FIX · DECLINED<small>Best candidate had only ${best.inliers} inliers (needs ${d.min_inliers}). For orbit determination, an honest "I don't know" beats a confident wrong answer.</small>`;
  }

  // mystery reveal
  if (mystery) {
    renderMeta(d, false);
    globe.pointsData([...S.photos]);
    let msg;
    if (fix) {
      S.score.rounds++;
      if (d.error_km < guessKm) { S.score.argus++; msg = `🤖 Argus wins this round, by <b>${fmtKm(guessKm - d.error_km)}</b>.`; }
      else { S.score.you++; msg = `🏆 You beat the machine! Argus missed by more than you did.`; }
      saveScore();
    } else {
      msg = `Argus declined this one. You were <b>${fmtKm(guessKm)}</b> off. No score change.`;
    }
    $("#guess-status").innerHTML = msg;
    renderScore();
  }

  const coords = $("#coords");
  const kpis = [];
  if (fix) {
    typeText(coords, fmtLatLon(d.pred_center[0], d.pred_center[1]));
    const width = Math.sqrt(d.area_km2);
    kpis.push(kpi("Off by", fmtKm(d.error_km), d.error_km < 10 ? "good" : "warn", "vs. true photo center"));
    kpis.push(kpi("Inliers", best.inliers, "", `of ${best.matches} matches`));
    kpis.push(kpi("Photo spans", `${fmt(width)} km`, "", `error ≈ ${fmt((100 * d.error_km) / width, 1)}% of width`));
    if (mystery) kpis.push(kpi("Your guess", fmtKm(guessKm), "vs", "off from truth"));
  } else {
    coords.textContent = "— no answer —";
    kpis.push(kpi("Best inliers", best.inliers, "warn", `threshold ${d.min_inliers}`));
    kpis.push(kpi("Candidates", d.candidates.length, "", "all rejected"));
    kpis.push(kpi("Truth", `${d.truth_center[0].toFixed(2)}, ${d.truth_center[1].toFixed(2)}`, "", "revealed"));
    if (mystery) kpis.push(kpi("Your guess", fmtKm(guessKm), "vs", "off from truth"));
  }
  $("#kpis").innerHTML = kpis.join("");
  $("#kpis").style.gridTemplateColumns = `repeat(${kpis.length}, 1fr)`;

  const total = d.timings.embed_ms + d.timings.search_ms + d.candidates.reduce((a, c) => a + c.match_ms, 0);
  $("#realtime").innerHTML = `Real pipeline time for this photo: <b>${fmt(total / 1000, 2)} s</b> on one GPU (embed ${fmt(d.timings.embed_ms)} ms, search ${fmt(d.timings.search_ms)} ms, 15 matches ${fmt(total - d.timings.embed_ms - d.timings.search_ms)} ms). The replay above is slowed down so you can watch it.`;
  $("#fromme").textContent = "";

  showCandidate(d.best);
  drawDetailMap();

  // globe
  const fp = [fpPath(d.truth_corners, "rgba(255,179,71,0.95)", 0.9, true)];
  if (fix) fp.push(fpPath(d.pred_corners, "rgba(94,242,255,1)", 1.1));
  paths.footprints = fp; refreshPaths();
  const [la, lo] = fix ? d.pred_center : d.truth_center;
  setRings([{ lat: la, lng: lo, color: fix ? "rgba(77,255,157,A)" : "rgba(255,179,71,A)", r: 3, speed: 1.8, period: 1100 }]);
  globe.arcsData([{ a0: d.nadir[0], b0: d.nadir[1], a1: d.truth_center[0], b1: d.truth_center[1] }]);
  setLabels([{ lat: d.nadir[0], lng: d.nadir[1], text: "ISS overhead", color: "rgba(177,140,255,0.95)", kind: "iss" }]);
  flyTo(la, lo, 0.3, 2000);

  setTimeout(() => res.scrollIntoView({ behavior: "smooth", block: "start" }), 250);
}

function kpi(label, value, cls, small) {
  return `<div class="kpi ${cls}"><span>${label}</span><b>${value}</b><small>${small}</small></div>`;
}

let lineAnim = 0;
function showCandidate(i) {
  const d = S.detail, c = d.candidates[i];
  S.selectedCand = i;
  document.querySelectorAll(".cand").forEach((e, j) => e.classList.toggle("viewing", j === i && i !== d.best));
  const a = S.aspect || 1.5, W = 100 * a + 110;
  const mv = $("#matchviz");
  mv.style.aspectRatio = String(W / 100);
  $("#mv-q").src = d.gape_img;
  $("#mv-q").style.width = `${(100 * a / W) * 100}%`;
  const slot = $("#mv-t");
  slot.style.width = `${(100 / W) * 100}%`;
  slot.replaceChildren(tileMosaic(c, 1));
  $("#mv-svg").setAttribute("viewBox", `0 0 ${W} 100`);
  $("#mv-r-label").textContent = i === d.best ? `best tile #${i + 1}` : `candidate #${i + 1}`;
  const pass = c.inliers >= d.min_inliers;
  $("#mv-caption").innerHTML = `${i === d.best ? "Best candidate" : `Candidate #${i + 1}`}: <b>${c.inliers}</b> inlier correspondences${c.lines.length < c.inliers ? ` (showing ${c.lines.length})` : ""}, tile center ${fmtKm(c.dist_to_truth_km)} from truth. ${pass ? "" : "Too few to trust. "}<span class="muted">Tap any candidate above to inspect it.</span>`;

  const svg = $("#mv-svg");
  svg.innerHTML = "";
  const NS = "http://www.w3.org/2000/svg";
  const ls = c.lines.map(([qx, qy, tx, ty], k) => {
    const hue = pass ? 170 + (k / Math.max(1, c.lines.length)) * 110 : 350;
    const ln = document.createElementNS(NS, "line");
    ln.setAttribute("stroke", `hsla(${hue},95%,65%,.85)`);
    const x1 = qx * 100 * a, y1 = qy * 100, x2 = 100 * a + 10 + tx * 100, y2 = ty * 100;
    ln.setAttribute("x1", x1); ln.setAttribute("y1", y1); ln.setAttribute("x2", x1); ln.setAttribute("y2", y1);
    svg.appendChild(ln);
    return { ln, x1, y1, x2, y2, delay: k * 18 };
  });
  const id = ++lineAnim, t0 = performance.now();
  const step = (t) => {
    if (id !== lineAnim) return;
    let more = false;
    for (const l of ls) {
      const k = Math.max(0, Math.min(1, (t - t0 - l.delay) / 500));
      if (k < 1) more = true;
      const e = 1 - Math.pow(1 - k, 3);
      l.ln.setAttribute("x2", l.x1 + (l.x2 - l.x1) * e);
      l.ln.setAttribute("y2", l.y1 + (l.y2 - l.y1) * e);
    }
    if (more) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);

  if (S.map && S.tileLayer) {
    S.tileLayer.setLatLngs(c.corners);
  }
}

function drawDetailMap() {
  const d = S.detail, fix = d.status === "fix";
  if (!S.map) {
    S.map = L.map("detail-map", { zoomControl: false, attributionControl: true, worldCopyJump: true });
    L.tileLayer(EOX("2021", "{z}", "{y}", "{x}"), { maxZoom: 14, attribution: EOX_ATTR }).addTo(S.map);
    L.control.zoom({ position: "bottomright" }).addTo(S.map);
    S.overlay = L.layerGroup().addTo(S.map);
  }
  S.overlay.clearLayers();
  const best = d.candidates[S.selectedCand ?? d.best];
  S.tileLayer = L.polygon(best.corners, { color: "#fff", weight: 1.2, opacity: 0.55, dashArray: "2 5", fill: false }).addTo(S.overlay);
  const truth = L.polygon(d.truth_corners, { color: "#ffb347", weight: 2.5, dashArray: "7 6", fillOpacity: 0.08 }).addTo(S.overlay);
  L.circleMarker(d.truth_center, { radius: 5, color: "#ffb347", fillColor: "#ffb347", fillOpacity: 1 }).addTo(S.overlay)
    .bindTooltip("truth", { permanent: true, direction: "left", className: "map-lbl" });
  let bounds = truth.getBounds();
  if (fix) {
    const pred = L.polygon(d.pred_corners, { color: "#5ef2ff", weight: 2.5, fillOpacity: 0.1 }).addTo(S.overlay);
    L.circleMarker(d.pred_center, { radius: 5, color: "#5ef2ff", fillColor: "#5ef2ff", fillOpacity: 1 }).addTo(S.overlay)
      .bindTooltip("Argus", { permanent: true, direction: "right", className: "map-lbl" });
    L.polyline([d.truth_center, d.pred_center], { color: "#fff", weight: 1.5, dashArray: "3 4" }).addTo(S.overlay);
    bounds = bounds.extend(pred.getBounds());
  } else {
    bounds = bounds.extend(S.tileLayer.getBounds());
  }
  if (S.mode === "mystery" && S.guess) {
    L.marker([S.guess.lat, S.guess.lng], { icon: L.divIcon({ html: "📍", className: "", iconSize: [24, 24], iconAnchor: [12, 24] }) }).addTo(S.overlay)
      .bindTooltip("you", { permanent: true, direction: "top", className: "map-lbl", offset: [0, -20] });
  }
  setTimeout(() => { S.map.invalidateSize(); S.map.fitBounds(bounds, { padding: [24, 24] }); }, 60);
}

/* ------------------------------------------------------------------ extras */
function fromMe() {
  const d = S.detail;
  if (!d) return;
  const target = d.pred_center || d.truth_center;
  const out = $("#fromme");
  if (!navigator.geolocation) { out.textContent = "Your browser doesn't share location."; return; }
  out.textContent = "Asking your browser… (nothing leaves this page)";
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      const km = haversine(pos.coords.latitude, pos.coords.longitude, target[0], target[1]);
      const min = km / ISS_KMS / 60;
      out.innerHTML = `📍 This spot is <b>${fmt(km)} km</b> from you. At ${ISS_KMS} km/s, the ISS would get there in about <b>${min < 1 ? "<1" : fmt(min)} min</b>.`;
    },
    () => { out.textContent = "Location permission denied. No worries, the ISS knows where you are anyway. 🛰️"; },
    { timeout: 8000 },
  );
}

function renderScore() {
  const sb = $("#scoreboard");
  if (!S.score.rounds) { sb.classList.add("hidden"); return; }
  sb.classList.remove("hidden");
  sb.innerHTML = `<div class="kicker" style="color:var(--violet)">BEAT THE AI · SCOREBOARD</div>
    <div style="display:flex;justify-content:space-around;margin-top:8px;text-align:center">
      <div><div class="big">${S.score.you}</div>you</div>
      <div><div class="big" style="color:var(--cyan)">${S.score.argus}</div>Argus</div>
      <div><div class="big" style="color:var(--muted)">${S.score.rounds}</div>rounds</div></div>`;
}

function initTicker() {
  const st = S.stats, fp = st.full_pipeline || {}, rec = st.recall || {};
  const bestR1 = Object.entries(rec).sort((a, b) => b[1]["1"] - a[1]["1"])[0];
  const items = [
    fp.median_error_km != null && `median localization error <b>${fmt(fp.median_error_km, 1)} km</b>`,
    fp.n_queries && `<b>${fmt(fp.n_queries)}</b> photos in the full benchmark run`,
    bestR1 && `top-1 retrieval recall up to <b>${fmt(bestR1[1]["1"], 1)}%</b> (${bestR1[0]})`,
    fp.median_latency_s && `<b>${fmt(fp.median_latency_s, 1)} s</b> per frame, end to end`,
    `GPS receivers used: <b>0</b>`,
    `ISS ground speed: <b>7.66 km/s</b>`,
    `refuses to guess below <b>${st.min_inliers}</b> inliers`,
    `retriever: <b>DINOv2 + SALAD</b> · matcher: <b>SIFT + LightGlue</b>`,
  ].filter(Boolean);
  $("#ticker-inner").innerHTML = [...items, ...items].map((s) => `<span>◆ ${s}</span>`).join("");
}

function initHood() {
  const st = S.stats, fp = st.full_pipeline || {};
  const tiles = Object.values(st.db_tiles || {}).reduce((a, b) => a + b, 0);
  $("#hood-kpis").innerHTML = [
    kpi("Median error", fp.median_error_km != null ? `${fmt(fp.median_error_km, 1)} km` : "—", "good", "matched fixes, full pipeline"),
    kpi("90th pct error", fp.p90_error_km != null ? `${fmt(fp.p90_error_km, 1)} km` : "—", "", "matched fixes"),
    kpi("Answers", fp.fix_rate != null ? `${fmt(fp.fix_rate)}%` : "—", "", `of ${fmt(fp.n_queries || 0)} photos; rest declined`),
    kpi("Latency", fp.median_latency_s != null ? `${fmt(fp.median_latency_s, 1)} s` : "—", "", "per frame, one GPU"),
    kpi("Reference map", fmt(tiles), "", "satellite tiles, 6 regions"),
    kpi("Descriptor", "2048-d", "", "per image / tile"),
  ].join("");
  $("#hood-kpis").style.gridTemplateColumns = "repeat(3, 1fr)";
  const rec = st.recall || {};
  const rows = Object.entries(rec).map(([r, v]) =>
    `<tr><td>${r}</td>${["1", "5", "10", "100"].map((k) => `<td class="bar"><i style="width:${0.7 * v[k]}%"></i><span>${fmt(v[k], 1)}%</span></td>`).join("")}</tr>`).join("");
  $("#recall-table").innerHTML = `<tr><th>REGION</th><th>R@1</th><th>R@5</th><th>R@10</th><th>R@100</th></tr>${rows}`;
}

function initCredit() {
  const links = AUTHOR.links.map((l) => ` · <a href="${l.href}" target="_blank" rel="noopener" style="color:var(--cyan)">${l.label}</a>`).join("");
  const c = el("p", "tiny", `Built by <b style="color:var(--text)">${AUTHOR.name}</b> for Argus at Carnegie Mellon${links}`);
  $("#modal-mission .card").appendChild(c);
  const idle = el("p", "tiny", `Built by <b style="color:var(--text)">${AUTHOR.name}</b>${links} · <button class="link" id="idle-mission" style="font-size:12px;color:var(--cyan)">what's this for?</button>`);
  document.querySelector('[data-state="idle"]').appendChild(idle);
  $("#idle-mission").onclick = () => $("#modal-mission").classList.remove("hidden");
}

function randomPhoto(fixOnly = false) {
  const pool = fixOnly ? S.photos.filter((p) => p.status === "fix") : S.photos;
  let p;
  do { p = pool[(Math.random() * pool.length) | 0]; } while (S.current && p.id === S.current.id && pool.length > 1);
  return p;
}

/* ------------------------------------------------------------------ wiring */
$("#btn-random").onclick = () => selectPhoto(randomPhoto(), "explore");
$("#btn-mystery").onclick = () => selectPhoto(randomPhoto(true), "mystery");
$("#btn-localize").onclick = localize;
$("#btn-skip").onclick = () => { S.skip = true; };
$("#btn-back").onclick = () => {
  if (S.running) S.skip = true;
  S.current = null; S.mode = "explore"; guessPin = null;
  resetRun(); globe.pointsData([...S.photos]); setState("idle");
  globe.pointOfView({ altitude: 2.2 }, 1400);
};
$("#btn-next").onclick = () => {
  if (S.mode === "mystery") selectPhoto(randomPhoto(true), "mystery");
  else selectPhoto(randomPhoto(), "explore");
};
$("#btn-fromme").onclick = fromMe;
$("#btn-mission").onclick = () => $("#modal-mission").classList.remove("hidden");
$("#btn-hood").onclick = () => $("#modal-hood").classList.remove("hidden");
document.querySelectorAll(".modal").forEach((m) => m.addEventListener("click", (e) => {
  if (e.target === m || e.target.hasAttribute("data-close")) m.classList.add("hidden");
}));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") document.querySelectorAll(".modal").forEach((m) => m.classList.add("hidden"));
  if (e.key === "Enter" && !$("#btn-localize").classList.contains("hidden") && !$("#btn-localize").disabled && $('[data-state="photo"]').classList.contains("on")) localize();
});

boot().catch((err) => {
  console.error(err);
  $("#boot-log").innerHTML += `\n<span style="color:var(--red)">boot failed: ${err.message}</span>`;
});
