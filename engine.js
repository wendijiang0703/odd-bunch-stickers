// Odd Bunch sticker station — browser engine.
// A port of the Python prototype (ai_packaging/) that runs entirely in the browser:
// OpenCV.js for the outline and shape features, a direct call to Gemini with the
// tester's own key, the same checks, limb attachment and label layout.
// Behaviour that matters (AI brief, schema, word lists, stickers, presets) is
// exported from the Python code by tools/build_site.py into data/.
"use strict";

const Engine = (() => {
  let CFG = null, PRESETS = {}, STICKERS = {};
  const MAX_INPUT = 1600, WORK_SIZE = 800, RECT_WORK_SIZE = 480, PAD = 0.35;

  // ------------------------------------------------------------------ setup
  async function loadCV() {
    if (window.cv && window.cv.Mat) return;
    if (window.cv instanceof Promise) { window.cv = await window.cv; return; }
    await new Promise(res => {
      const t = setInterval(() => { if (window.cv && window.cv.Mat) { clearInterval(t); res(); } }, 100);
      if (window.cv) window.cv.onRuntimeInitialized = () => { clearInterval(t); res(); };
    });
  }

  async function init() {
    const [cfg, presets, samples] = await Promise.all(
      ["data/config.json", "data/presets.json", "data/samples.json"].map(u => fetch(u).then(r => r.json())));
    CFG = cfg; PRESETS = presets;
    await Promise.all(Object.keys(CFG.props).map(t => new Promise((res, rej) => {
      const im = new Image(); im.onload = () => { STICKERS[t] = im; res(); }; im.onerror = rej; im.src = `stickers/${t}.png`;
    })));
    await loadCV();
    try { await document.fonts.load("800 40px Nunito"); await document.fonts.load("italic 30px Lora"); } catch (e) { /* system fonts */ }
    return {samples, props: Object.fromEntries(Object.entries(CFG.props).map(([t, v]) => [t, v.desc]))};
  }

  // ------------------------------------------------------------------ helpers
  const canvas = (w, h) => { const c = document.createElement("canvas"); c.width = Math.max(1, Math.round(w)); c.height = Math.max(1, Math.round(h)); return c; };
  const r3 = v => Math.round(v * 1000) / 1000;
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const ellipseKernel = k => cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(k, k));

  function loadImage(src) {
    return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = () => rej(new Error("That file is not an image the browser can read (try JPG or PNG).")); im.src = src; });
  }

  // photo -> RGB cv.Mat, at most MAX_INPUT on the long side
  function photoMat(img) {
    const s = Math.min(1, MAX_INPUT / Math.max(img.naturalWidth || img.width, img.naturalHeight || img.height));
    const c = canvas((img.naturalWidth || img.width) * s, (img.naturalHeight || img.height) * s);
    c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
    const rgba = cv.imread(c), rgb = new cv.Mat();
    cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB); rgba.delete();
    return rgb;
  }

  function largestBlob(mask) {   // returns {mask, contour} (caller deletes); fills holes
    const contours = new cv.MatVector(), hier = new cv.Mat();
    cv.findContours(mask, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE);
    let best = -1, bestA = 0;
    for (let i = 0; i < contours.size(); i++) { const a = cv.contourArea(contours.get(i)); if (a > bestA) { bestA = a; best = i; } }
    if (best < 0) { contours.delete(); hier.delete(); return {mask: null, contour: null}; }
    const filled = cv.Mat.zeros(mask.rows, mask.cols, cv.CV_8U);
    cv.drawContours(filled, contours, best, new cv.Scalar(255), cv.FILLED);
    const contour = contours.get(best).clone();
    contours.delete(); hier.delete();
    return {mask: filled, contour};
  }

  function contourAreas(mask) {
    const contours = new cv.MatVector(), hier = new cv.Mat();
    cv.findContours(mask, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    const a = []; for (let i = 0; i < contours.size(); i++) a.push(cv.contourArea(contours.get(i)));
    contours.delete(); hier.delete(); return a.sort((x, y) => y - x);
  }

  // ------------------------------------------------------------------ isolate (segment.py)
  function backgroundDistanceMask(img) {
    const h = img.rows, w = img.cols, b = Math.max(4, Math.floor(0.04 * Math.min(h, w)));
    const lab = new cv.Mat(); cv.cvtColor(img, lab, cv.COLOR_RGB2Lab);
    const L = lab.data, n = h * w;
    const border = [[], [], []];
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (y < b || y >= h - b || x < b || x >= w - b) { const i = (y * w + x) * 3; border[0].push(L[i]); border[1].push(L[i + 1]); border[2].push(L[i + 2]); }
    }
    const median = a => { a.sort((p, q) => p - q); return a[a.length >> 1]; };
    const bg = border.map(median);
    const dist = new cv.Mat(h, w, cv.CV_32F), D = dist.data32F;
    const shadow = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const dL = L[i * 3] - bg[0], da = L[i * 3 + 1] - bg[1], db = L[i * 3 + 2] - bg[2];
      D[i] = Math.sqrt((0.35 * dL) ** 2 + da * da + db * db);
      shadow[i] = (Math.sqrt(da * da + db * db) < 7 && dL < -6) ? 1 : 0;
    }
    lab.delete();
    cv.GaussianBlur(dist, dist, new cv.Size(5, 5), 0);
    const d8 = new cv.Mat(h, w, cv.CV_8U);
    for (let i = 0; i < n; i++) d8.data[i] = Math.min(255, Math.max(0, D[i] * 3));
    dist.delete();
    const tmp = new cv.Mat();
    const otsu = cv.threshold(d8, tmp, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU); tmp.delete();
    const thresh = Math.max(otsu, 36);
    const mask = new cv.Mat(h, w, cv.CV_8U);
    let on = 0, sh = 0;
    for (let i = 0; i < n; i++) { const m = d8.data[i] > thresh; mask.data[i] = m ? 255 : 0; if (m) { on++; if (shadow[i]) sh++; } }
    d8.delete();
    if (sh < 0.5 * on) for (let i = 0; i < n; i++) if (shadow[i]) mask.data[i] = 0;  // soft shadows on a white board
    const k = Math.max(3, Math.floor(Math.min(h, w) / 150) | 1);
    const k1 = ellipseKernel(k), k3 = ellipseKernel(3 * k);
    cv.morphologyEx(mask, mask, cv.MORPH_OPEN, k1); cv.morphologyEx(mask, mask, cv.MORPH_CLOSE, k3);
    k1.delete(); k3.delete();
    return mask;
  }

  function plausible(mask, contour) {
    if (!contour) return [false, "nothing found"];
    const h = mask.rows, w = mask.cols, frac = cv.contourArea(contour) / (h * w);
    if (frac < 0.005) return [false, `blob too small (${frac.toFixed(3)} of frame)`];
    if (frac > 0.85) return [false, `blob fills the frame (${frac.toFixed(2)})`];
    const p = contour.data32S; let edge = 0; const np = p.length / 2;
    for (let i = 0; i < p.length; i += 2) if (p[i] <= 1 || p[i + 1] <= 1 || p[i] >= w - 2 || p[i + 1] >= h - 2) edge++;
    if (edge / np > 0.05) return [false, `blob runs along the image edge (${Math.round(100 * edge / np)}%)`];
    return [true, ""];
  }

  function grabcut(img, seed, rect, iters) {
    const gc = new cv.Mat(img.rows, img.cols, cv.CV_8U, new cv.Scalar(cv.GC_BGD));
    const bgd = new cv.Mat(), fgd = new cv.Mat();
    if (seed) {
      const k = Math.max(3, Math.floor(Math.min(img.rows, img.cols) / 60) | 1), ker = ellipseKernel(k);
      const er = new cv.Mat(), di = new cv.Mat();
      cv.erode(seed, er, ker, new cv.Point(-1, -1), 2); cv.dilate(seed, di, ker, new cv.Point(-1, -1), 3);
      for (let i = 0; i < gc.data.length; i++) {
        gc.data[i] = di.data[i] === 0 ? cv.GC_BGD : er.data[i] ? cv.GC_FGD : seed.data[i] ? cv.GC_PR_FGD : cv.GC_PR_BGD;
      }
      er.delete(); di.delete(); ker.delete();
      cv.grabCut(img, gc, new cv.Rect(0, 0, 0, 0), bgd, fgd, iters, cv.GC_INIT_WITH_MASK);
    } else {
      cv.grabCut(img, gc, rect, bgd, fgd, iters, cv.GC_INIT_WITH_RECT);
    }
    for (let i = 0; i < gc.data.length; i++) gc.data[i] = (gc.data[i] === cv.GC_FGD || gc.data[i] === cv.GC_PR_FGD) ? 255 : 0;
    bgd.delete(); fgd.delete();
    return gc;
  }

  function resizeMask(m, w, h) {
    const out = new cv.Mat(); cv.resize(m, out, new cv.Size(w, h), 0, 0, cv.INTER_LINEAR);
    cv.threshold(out, out, 127, 255, cv.THRESH_BINARY); return out;
  }

  function segment(photo) {
    const H = photo.rows, W = photo.cols, scale = Math.min(1, WORK_SIZE / Math.max(H, W));
    const small = new cv.Mat();
    cv.resize(photo, small, new cv.Size(Math.round(W * scale), Math.round(H * scale)), 0, 0, cv.INTER_AREA);
    const sh = small.rows, sw = small.cols;
    const raw = backgroundDistanceMask(small);
    let {mask: rough, contour: c} = largestBlob(raw);
    let [ok, why] = plausible(raw, c);
    let mask, method, note = "";
    if (ok) {
      method = "background-colour + grabcut";
      const g = grabcut(small, rough, null, 4), r = largestBlob(g); g.delete();
      if (r.contour && cv.contourArea(r.contour) / Math.max(cv.contourArea(c), 1) > 0.7 &&
          cv.contourArea(r.contour) / Math.max(cv.contourArea(c), 1) < 1.3) { mask = r.mask; rough.delete(); }
      else { mask = rough; method = "background-colour"; if (r.mask) r.mask.delete(); }
      if (r.contour) r.contour.delete();
    } else {
      note = `quick mask rejected: ${why}`;
      method = "grabcut (centre rectangle)";
      if (rough) rough.delete();
      const rr = Math.min(1, RECT_WORK_SIZE / Math.max(sh, sw)), tiny = new cv.Mat();
      cv.resize(small, tiny, new cv.Size(Math.round(sw * rr), Math.round(sh * rr)), 0, 0, cv.INTER_AREA);
      const mx = Math.floor(0.06 * tiny.cols), my = Math.floor(0.06 * tiny.rows);
      let g = grabcut(tiny, null, new cv.Rect(mx, my, tiny.cols - 2 * mx, tiny.rows - 2 * my), 5); tiny.delete();
      if (rr < 1) { const up = resizeMask(g, sw, sh); g.delete(); g = up; }
      const r = largestBlob(g); g.delete();
      const [ok2, why2] = plausible(r.mask || raw, r.contour);
      if (!ok2) {
        small.delete(); raw.delete(); if (c) c.delete();
        if (r.mask) r.mask.delete(); if (r.contour) r.contour.delete();
        throw new Error(`Could not find a single vegetable (${why}; grabcut: ${why2}). Use a plain white board and keep the veg away from the edges.`);
      }
      mask = r.mask; r.contour.delete();
    }
    if (c) c.delete();
    const warnings = [];
    if (method.startsWith("grabcut (centre")) warnings.push("The background is not plain, so the outline may include some background. A plain white board works best.");
    const areas = contourAreas(raw); raw.delete(); small.delete();
    if (areas.length > 1 && areas[1] > 0.3 * areas[0]) warnings.push("More than one object in the photo; using the largest. Photograph one veg at a time.");
    if (scale < 1) { const up = resizeMask(mask, W, H); mask.delete(); mask = up; }
    const r = largestBlob(mask); mask.delete();
    const bb = cv.boundingRect(r.contour);
    if (bb.x <= 1 || bb.y <= 1 || bb.x + bb.width >= W - 1 || bb.y + bb.height >= H - 1)
      warnings.push("The veg touches the edge of the photo, so part of it may be cut off. Move it to the middle.");
    return {mask: r.mask, contour: r.contour, bbox: [bb.x, bb.y, bb.width, bb.height], method, note, warnings};
  }

  // ------------------------------------------------------------------ shape features (features.py)
  function colourName(rgb) {
    const px = cv.matFromArray(1, 1, cv.CV_8UC3, rgb.map(Math.round)), hsv = new cv.Mat();
    cv.cvtColor(px, hsv, cv.COLOR_RGB2HSV); const [h0, s, v] = hsv.data; px.delete(); hsv.delete();
    const h = h0 * 2;
    if (v < 50) return "very dark";
    if (s < 40) return v > 170 ? "pale / white" : "grey-brown";
    if (h < 15 || h >= 345) return "red";
    if (h < 40) return s > 140 ? "orange" : "brown / tan";
    if (h < 70) return s > 120 ? "yellow" : "tan";
    if (h < 170) return "green";
    if (h < 260) return "blue";
    return "purple";
  }

  function faceAnchor(maskCrop) {
    const dist = new cv.Mat(); cv.distanceTransform(maskCrop, dist, cv.DIST_L2, 5);
    const mm = cv.minMaxLoc(dist); dist.delete();
    return [mm.maxLoc.x, mm.maxLoc.y, mm.maxVal];
  }

  function mirrorIoU(mask, cx, cy, angleDeg) {
    // work on a small copy: the result is a ratio
    const s = Math.min(1, 300 / Math.max(mask.rows, mask.cols)), m = new cv.Mat();
    cv.resize(mask, m, new cv.Size(Math.max(1, Math.round(mask.cols * s)), Math.max(1, Math.round(mask.rows * s))), 0, 0, cv.INTER_NEAREST);
    const side = Math.floor(2 * Math.hypot(m.rows, m.cols)) + 4;
    const M = cv.getRotationMatrix2D(new cv.Point(cx * s, cy * s), angleDeg, 1.0);
    M.data64F[2] += side / 2 - cx * s; M.data64F[5] += side / 2 - cy * s;
    const rot = new cv.Mat(); cv.warpAffine(m, rot, M, new cv.Size(side, side), cv.INTER_NEAREST); m.delete(); M.delete();
    const d = rot.data; let iMain = 0, uMain = 0, iCross = 0, uCross = 0;
    for (let y = 0; y < side; y++) for (let x = 0; x < side; x++) {
      const a = d[y * side + x] > 0, b = d[(side - 1 - y) * side + x] > 0, c = d[y * side + (side - 1 - x)] > 0;
      if (a || b) { uMain++; if (a && b) iMain++; }
      if (a || c) { uCross++; if (a && c) iCross++; }
    }
    rot.delete();
    return [uMain ? iMain / uMain : 0, uCross ? iCross / uCross : 0];
  }

  function shapeFeatures(photo, seg) {
    const H = photo.rows, W = photo.cols, c = seg.contour, [x, y, w, h] = seg.bbox;
    const area = cv.contourArea(c), perim = cv.arcLength(c, true);
    const smooth = new cv.Mat(); cv.approxPolyDP(c, smooth, 0.002 * perim, true);
    const perimS = cv.arcLength(smooth, true);
    const rr = cv.minAreaRect(c), longSide = Math.max(rr.size.width, rr.size.height), shortSide = Math.max(Math.min(rr.size.width, rr.size.height), 1e-6);
    const hull = new cv.Mat(); cv.convexHull(c, hull, false, true); const hullArea = cv.contourArea(hull); hull.delete();
    let depths = [];
    try {
      const hi = new cv.Mat(); cv.convexHull(smooth, hi, false, false);
      if (hi.rows > 3) { const df = new cv.Mat(); cv.convexityDefects(smooth, hi, df); for (let i = 0; i < df.data32S.length; i += 4) depths.push(df.data32S[i + 3] / 256); df.delete(); }
      hi.delete();
    } catch (e) { /* self-intersecting outline: no defects */ }
    smooth.delete();
    depths.sort((a, b) => b - a);
    const sig = depths.filter(d => d > 0.05 * longSide);
    const mo = cv.moments(c), cx = mo.m10 / mo.m00, cy = mo.m01 / mo.m00;
    const theta = 0.5 * Math.atan2(2 * mo.mu11, mo.mu20 - mo.mu02) * 180 / Math.PI;
    const [symMain, symCross] = mirrorIoU(seg.mask, cx, cy, theta);
    const inner = new cv.Mat(), k7 = cv.Mat.ones(7, 7, cv.CV_8U); cv.erode(seg.mask, inner, k7); k7.delete();
    const mean = cv.mean(photo, cv.countNonZero(inner) ? inner : seg.mask); inner.delete();
    const crop = seg.mask.roi(new cv.Rect(x, y, w, h)); const [ax, ay, ar] = faceAnchor(crop); crop.delete();
    const hex = "#" + [mean[0], mean[1], mean[2]].map(v => Math.round(v).toString(16).padStart(2, "0")).join("");
    const nd = CFG.no_data;
    return {
      image_size: [W, H], segmentation_method: seg.method, area_px: Math.round(area),
      bbox: [x, y, w, h], elongation: r3(longSide / shortSide), orientation_deg: Math.round(theta * 10) / 10,
      circularity: r3(Math.min(1, 4 * Math.PI * area / perimS ** 2)), solidity: r3(hullArea ? area / hullArea : 0),
      bumps: sig.length, max_dent_depth_ratio: r3(depths.length ? depths[0] / longSide : 0),
      symmetry_main_axis: r3(symMain), symmetry_cross_axis: r3(symCross), symmetry_score: r3(Math.max(symMain, symCross)),
      mean_colour_hex: hex, colour_name: colourName([mean[0], mean[1], mean[2]]),
      face_anchor: {x: r3(ax / w), y: r3(ay / h), radius_over_short_side: r3(ar / Math.min(w, h))},
      size: {length_mm: nd, width_mm: nd, area_cm2: nd},
    };
  }

  const PROMPT_KEYS = ["elongation", "orientation_deg", "circularity", "solidity", "bumps", "max_dent_depth_ratio",
                       "symmetry_main_axis", "symmetry_cross_axis", "colour_name", "face_anchor", "size"];

  // ------------------------------------------------------------------ checks (safety.py)
  const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  let WORD_RE = null, NAME_RE = null;
  function textProblems(t) {
    WORD_RE = WORD_RE || new RegExp(`\\b(${CFG.safety.banned_words.map(esc).join("|")})\\b`, "gi");
    NAME_RE = NAME_RE || new RegExp(`\\b(${CFG.safety.protected_names.map(esc).join("|")})\\b`, "gi");
    return [...t.matchAll(WORD_RE)].map(m => `uses a banned word '${m[0]}'`)
      .concat([...t.matchAll(NAME_RE)].map(m => `references a real person or existing character '${m[0]}'`));
  }
  const isNum = v => typeof v === "number" && isFinite(v);

  function validate(d) {
    if (!d || typeof d !== "object" || Array.isArray(d)) return [null, ["response is not a JSON object"]];
    const TF = ["resembles", "object_name", "tagline"];
    let problems = TF.filter(k => typeof d[k] !== "string" || !d[k].trim()).map(k => `missing or empty '${k}'`);
    if (problems.length) return [null, problems];
    const clean = Object.fromEntries(TF.map(k => [k, d[k].trim()]));
    clean.veg_type = String(d.veg_type || CFG.no_data).trim().toLowerCase();
    if (clean.object_name.length > 20) problems.push("object_name longer than 20 characters");
    if (clean.tagline.length > 60) problems.push("tagline longer than 60 characters");
    for (const k of TF) problems.push(...textProblems(clean[k]).map(p => `${k} ${p}`));
    const S = CFG.safety, types = Object.keys(CFG.props);
    if (!Array.isArray(d.props) || !d.props.length) return [null, problems.concat(["'props' must be a non-empty list"])];
    if (d.props.length > S.max_props) problems.push(`too many props (${d.props.length} > ${S.max_props})`);
    const out = [];
    d.props.slice(0, S.max_props).forEach((p, i) => {
      if (!p || typeof p !== "object") return problems.push(`props[${i}] is not an object`);
      if (!types.includes(p.type)) return problems.push(`props[${i}].type '${p.type}' is not in the sticker library`);
      const bad = ["x", "y", "size"].filter(k => !isNum(p[k]));
      if (bad.length) return problems.push(`props[${i}] (${p.type}) needs numbers for ${bad.join(", ")}`);
      if (p.x < S.xy_range[0] || p.x > S.xy_range[1] || p.y < S.xy_range[0] || p.y > S.xy_range[1])
        problems.push(`props[${i}] (${p.type}) position (${p.x.toFixed(2)}, ${p.y.toFixed(2)}) is too far outside the veg`);
      if (p.size < S.size_range[0] || p.size > S.size_range[1]) problems.push(`props[${i}] (${p.type}) size ${p.size.toFixed(2)} must be ${S.size_range[0]}..${S.size_range[1]}`);
      out.push({type: p.type, x: r3(p.x), y: r3(p.y), size: r3(p.size), angle: Math.round(clamp(isNum(p.angle) ? p.angle : 0, -180, 180) * 10) / 10, flip: !!p.flip});
    });
    if (problems.length) return [null, problems];
    const res = {...clean, props: out};
    if (Array.isArray(d.ideas)) res.ideas = d.ideas.slice(0, 5).map(s => String(s).slice(0, 120));
    return [res, []];
  }

  // ------------------------------------------------------------------ offline rules (match_rules.py)
  function ruleMatch(f) {
    let key, reason;
    if (f.elongation >= 2.6) [key, reason] = ["striker", `long and thin (elongation ${f.elongation} >= 2.6)`];
    else if (f.max_dent_depth_ratio >= 0.12 && f.bumps <= 2) [key, reason] = ["hugger", `one deep dent between two lobes`];
    else if (f.bumps >= 3) [key, reason] = ["grandad", `knobbly (${f.bumps} bumps)`];
    else if (f.circularity >= 0.82 && f.elongation < 1.3) [key, reason] = ["smiley", `round (circularity ${f.circularity})`];
    else if (f.symmetry_score < 0.8) [key, reason] = ["cheeky", `lopsided (symmetry ${f.symmetry_score})`];
    else [key, reason] = ["sleepy", "smooth oval, no stand-out feature"];
    const a = CFG.rules[key], [w, h] = [f.bbox[2], f.bbox[3]], L = Math.max(w, h), an = f.face_anchor;
    const cx = an.x * w, cy = an.y * h, r = an.radius_over_short_side * Math.min(w, h);
    const P = (t, dx, dy, wr) => ({type: t, x: r3((cx + dx * r) / w), y: r3((cy + dy * r) / h), size: r3(clamp(wr * r / L, 0.03, 1)), angle: 0, flip: false});
    const ew = a.eyes === "eye_dot" ? 0.3 : 0.45;
    const props = [P(a.eyes, -0.35, -0.22, ew), P(a.eyes, 0.35, -0.22, ew), P(a.mouth, 0, 0.3, 0.5)];
    if (a.extra === "blush") props.push(P("blush", -0.58, 0.1, 0.35), P("blush", 0.58, 0.1, 0.35));
    const tagline = a.taglines[f.area_px % a.taglines.length];
    return [{resembles: a.resembles, object_name: a.name, tagline, veg_type: CFG.no_data, props}, reason];
  }

  // ------------------------------------------------------------------ Gemini (match_ai.py)
  const KEY_STORE = "oddbunch_gemini_key";
  const getKey = () => { try { return localStorage.getItem(KEY_STORE) || ""; } catch (e) { return ""; } };

  class Unavailable extends Error {}
  class Busy extends Unavailable {}

  async function geminiRequest(key, model, body) {
    const ctl = new AbortController(), t = setTimeout(() => ctl.abort(), CFG.gemini.timeout_s * 1000);
    let r;
    try {
      r = await fetch("https://generativelanguage.googleapis.com/v1beta/interactions", {
        method: "POST", signal: ctl.signal,
        headers: {"content-type": "application/json", "x-goog-api-key": key},
        body: JSON.stringify({model, ...body, store: false}),
      });
    } catch (e) {
      throw new Unavailable(e.name === "AbortError" ? `Gemini took longer than ${CFG.gemini.timeout_s} s and timed out; try again`
                                                      : "no connection to Gemini (check the internet)");
    } finally { clearTimeout(t); }
    let j = {}; try { j = await r.json(); } catch (e) { /* not JSON */ }
    if (!r.ok) {
      const err = j.error || {}, reason = (err.details || []).map(d => d.reason).find(Boolean) || err.status || "";
      const msg = `Gemini error ${r.status}${reason ? " " + reason : ""}: ${err.message || r.statusText}`;
      const hints = {API_KEY_INVALID: "Google says this API key is not valid. Copy it again from aistudio.google.com/apikey",
                     ACCESS_TOKEN_TYPE_UNSUPPORTED: "Google says this API key is not valid. Copy it again from aistudio.google.com/apikey",
                     PERMISSION_DENIED: "this key has no permission for the Gemini API",
                     NOT_FOUND: `the model '${model}' was not found for this key`};
      const text = hints[reason] ? `${hints[reason]} (${msg})` : msg;
      if (r.status === 429 || r.status === 503 || reason === "RESOURCE_EXHAUSTED" || reason === "UNAVAILABLE") throw new Busy(text);
      throw new Unavailable(text);
    }
    return (j.steps || []).filter(s => s.type === "model_output").flatMap(s => s.content || []).map(c => c.text || "").join("");
  }

  async function geminiCall(key, body) {
    const models = [...new Set([CFG.gemini.model, ...CFG.gemini.backups])], busy = [];
    for (const m of models) {
      try { return [await geminiRequest(key, m, body), m]; }
      catch (e) { if (!(e instanceof Busy)) throw e; busy.push(m); }
    }
    throw new Unavailable(`all Gemini models are busy right now (${busy.join(", ")}); try again in a minute`);
  }

  async function checkKey(key) {
    try {
      const [txt, model] = await geminiCall(key, {input: [{type: "text", text: "Reply with the single word OK."}]});
      return {ok: true, message: `Gemini (${model}) replied: '${txt.trim().slice(0, 40)}'`};
    } catch (e) { return {ok: false, error: e.message}; }
  }

  function cropJpegB64(photo, bbox) {
    const [x, y, w, h] = bbox, roi = photo.roi(new cv.Rect(x, y, w, h));
    const s = Math.min(1, 640 / Math.max(w, h)), small = new cv.Mat();
    cv.resize(roi, small, new cv.Size(Math.round(w * s), Math.round(h * s)), 0, 0, cv.INTER_AREA); roi.delete();
    const rgba = new cv.Mat(); cv.cvtColor(small, rgba, cv.COLOR_RGB2RGBA); small.delete();
    const c = canvas(rgba.cols, rgba.rows); cv.imshow(c, rgba); rgba.delete();
    return c.toDataURL("image/jpeg", 0.9).split(",")[1];
  }

  async function match(photo, seg, f, name, offline) {
    const t0 = performance.now(), key = getKey();
    const fallback = (reason, attempts = []) => { const [ch, why] = ruleMatch(f); return {character: ch, source: "rules", fallback_reason: reason, rules_reason: why, attempts, seconds: (performance.now() - t0) / 1000}; };
    if (offline) return fallback("offline mode");
    if (!key) {
      const preset = PRESETS[name];
      if (preset && !name.startsWith("_")) { const [c] = validate(preset); if (c) return {character: c, source: "preset", attempts: [], seconds: (performance.now() - t0) / 1000}; }
      return fallback("no Gemini key set (click the AI lamp)");
    }
    const image = cropJpegB64(photo, seg.bbox), feats = Object.fromEntries(PROMPT_KEYS.map(k => [k, f[k]]));
    const attempts = []; let problems = null;
    for (let n = 1; n <= 2; n++) {
      let text = "Shape measurements from OpenCV (context only; trust the photo for what it looks like):\n" + JSON.stringify(feats, null, 1) +
                 "\n\nWrite three ideas, pick the funniest or cutest pose the shape supports, and return the sticker plan.";
      if (problems) text += "\n\nA previous answer was rejected for these reasons, so fix them: " + problems.join("; ");
      const a0 = performance.now(); let raw, model;
      try {
        [raw, model] = await geminiCall(key, {
          system_instruction: CFG.system_prompt,
          input: [{type: "image", data: image, mime_type: "image/jpeg"}, {type: "text", text}],
          response_format: {type: "text", mime_type: "application/json", schema: CFG.schema},
          generation_config: {thinking_level: CFG.gemini.thinking},
        });
      } catch (e) { attempts.push({attempt: n, error: e.message}); return fallback(e.message, attempts); }
      const rec = {attempt: n, model, seconds: (performance.now() - a0) / 1000, raw_text: raw};
      let ch = null;
      try { [ch, problems] = validate(JSON.parse(raw)); } catch (e) { problems = ["invalid JSON"]; }
      if (ch) { attempts.push(rec); return {character: ch, source: "gemini", model, attempts, seconds: Math.round((performance.now() - t0) / 100) / 10}; }
      rec.problems = problems; attempts.push(rec);
    }
    return fallback("gemini answer failed the checks twice: " + (problems || []).join("; "), attempts);
  }

  // ------------------------------------------------------------------ limbs (pose.py)
  const ANCHORS = {arm: [-145 / 320, 0], arm_bent: [-108 / 240, 80 / 240], leg: [-20 / 150, -140 / 150]};
  const ARMS = ["arm", "arm_bent"], MIN_ARM = 0.16, MIN_LEG = 0.11;

  function placeAnchor(p, ax0, ay0, w, h, L) {
    const [ax, ay] = ANCHORS[p.type], width = p.size * L;
    let ox = ax * width; const oy = ay * width; if (p.flip) ox = -ox;
    const t = (p.angle || 0) * Math.PI / 180;
    const rx = ox * Math.cos(t) - oy * Math.sin(t), ry = ox * Math.sin(t) + oy * Math.cos(t);
    p.x = r3((ax0 - rx) / w); p.y = r3((ay0 - ry) / h);
  }

  function attachLimbs(ch, m, w, h) {   // m: Uint8Array mask of the veg box (w x h)
    const L = Math.max(w, h), out = {...ch, props: ch.props.map(p => ({...p}))};
    const at = (x, y) => m[y * w + x] > 0;
    const edgeX = (row, side) => {
      const step = Math.max(1, Math.floor(h / 50));
      for (let d = 0; d < h; d += step) for (const r of [row - d, row + d]) {
        if (r < 0 || r >= h) continue;
        if (side === "left") { for (let x = 0; x < w; x++) if (at(x, r)) return [x, r]; }
        else { for (let x = w - 1; x >= 0; x--) if (at(x, r)) return [x, r]; }
      }
      return [side === "left" ? 0 : w - 1, row];
    };
    const bottomY = col => {
      const step = Math.max(1, Math.floor(w / 50));
      for (let d = 0; d < w; d += step) for (const c of [col - d, col + d]) {
        if (c < 0 || c >= w) continue;
        for (let y = h - 1; y >= 0; y--) if (at(c, y)) return [y, c];
      }
      return [h - 1, col];
    };
    for (const p of out.props) { if (ARMS.includes(p.type)) p.size = Math.max(p.size, MIN_ARM); else if (p.type === "leg") p.size = Math.max(p.size, MIN_LEG); }
    for (const p of out.props) {
      if (!ARMS.includes(p.type)) continue;
      const side = p.x < 0.5 ? "left" : "right";
      if ((side === "left") !== !!p.flip) { p.flip = side === "left"; p.angle = -(p.angle || 0); }
      const [ex, ey] = edgeX(Math.floor(clamp(p.y, 0.05, 0.95) * (h - 1)), side);
      placeAnchor(p, ex + 0.02 * w * (side === "left" ? 1 : -1), ey, w, h, L);
    }
    const legs = out.props.filter(p => p.type === "leg").sort((a, b) => a.x - b.x);
    if (legs.length) {
      let xs = legs.map(p => clamp(p.x, 0.1, 0.9));
      if (legs.length === 2 && xs[1] - xs[0] < 0.2) { const mid = (xs[0] + xs[1]) / 2; xs = [mid - 0.12, mid + 0.12]; }
      legs.forEach((p, i) => {
        const [by, bc] = bottomY(Math.floor(xs[i] * (w - 1)));
        p.flip = legs.length === 2 && i === 0;
        placeAnchor(p, bc, by - 0.02 * h, w, h, L);
      });
    }
    return out;
  }

  // ------------------------------------------------------------------ render (render.py)
  const FONT = {display: '"Nunito", "Arial Rounded MT Bold", Arial, sans-serif', italic: 'Georgia, "Lora", serif', body: 'Arial, "Helvetica Neue", sans-serif'};

  function whiteEdge(src, px) {
    const pad = px + 2, c = canvas(src.width + 2 * pad, src.height + 2 * pad);
    c.getContext("2d").drawImage(src, pad, pad);
    const m = cv.imread(c), ch = new cv.MatVector(); cv.split(m, ch);
    const a = ch.get(3), k = ellipseKernel(2 * px + 1); cv.dilate(a, a, k); k.delete();
    const out = canvas(c.width, c.height), ctx = out.getContext("2d"), img = ctx.createImageData(c.width, c.height);
    for (let i = 0; i < a.data.length; i++) { img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = 255; img.data[i * 4 + 3] = a.data[i]; }
    a.delete(); ch.delete(); m.delete();
    ctx.putImageData(img, 0, 0); ctx.drawImage(c, 0, 0);
    return out;
  }

  function sticker(p, L) {
    const src = STICKERS[p.type], width = Math.max(8, Math.floor(p.size * L)), height = Math.max(1, Math.floor(src.height * width / src.width));
    let c = canvas(width, height); const x = c.getContext("2d");
    if (p.flip) { x.translate(width, 0); x.scale(-1, 1); }
    x.drawImage(src, 0, 0, width, height);
    c = whiteEdge(c, Math.max(2, Math.floor(width / 30)));
    const a = (p.angle || 0) * Math.PI / 180;
    if (Math.abs(p.angle || 0) > 0.5) {
      const W = Math.abs(c.width * Math.cos(a)) + Math.abs(c.height * Math.sin(a)), H = Math.abs(c.width * Math.sin(a)) + Math.abs(c.height * Math.cos(a));
      const r = canvas(W, H), rx = r.getContext("2d");
      rx.translate(r.width / 2, r.height / 2); rx.rotate(a); rx.drawImage(c, -c.width / 2, -c.height / 2);
      c = r;
    }
    return c;
  }

  function compose(veg, w, h, ch) {   // veg: canvas of the veg box (or null)
    const pad = Math.round(PAD * Math.max(w, h)), L = Math.max(w, h);
    const c = canvas(w + 2 * pad, h + 2 * pad), x = c.getContext("2d");
    const put = list => list.forEach(p => { const s = sticker(p, L); x.drawImage(s, Math.round(pad + p.x * w - s.width / 2), Math.round(pad + p.y * h - s.height / 2)); });
    put(ch.props.filter(p => CFG.props[p.type].behind));
    if (veg) x.drawImage(veg, pad, pad);
    put(ch.props.filter(p => !CFG.props[p.type].behind));
    return {canvas: c, pad};
  }

  function alphaBox(c) {
    const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
    let x0 = c.width, y0 = c.height, x1 = -1, y1 = -1;
    for (let y = 0; y < c.height; y++) for (let x = 0; x < c.width; x++) if (d[(y * c.width + x) * 4 + 3] > 8) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    return x1 < 0 ? [0, 0, c.width, c.height] : [x0, y0, x1 + 1, y1 + 1];
  }

  function cropCanvas(c, [x0, y0, x1, y1]) { const o = canvas(x1 - x0, y1 - y0); o.getContext("2d").drawImage(c, -x0, -y0); return o; }

  function trim(c, margin = 0.03) {
    const [x0, y0, x1, y1] = alphaBox(c), m = Math.floor(margin * Math.max(x1 - x0, y1 - y0));
    return cropCanvas(c, [Math.max(0, x0 - m), Math.max(0, y0 - m), Math.min(c.width, x1 + m), Math.min(c.height, y1 + m)]);
  }

  function fitFont(ctx, text, family, weight, maxSize, maxW, minSize = 18) {
    for (let s = maxSize; s > minSize; s -= 2) { ctx.font = `${weight} ${s}px ${family}`; if (ctx.measureText(text).width <= maxW) return s; }
    ctx.font = `${weight} ${minSize}px ${family}`; return minSize;
  }

  function wrap(ctx, text, maxW) {
    const lines = []; let cur = "";
    for (const word of text.split(/\s+/)) { const t = (cur + " " + word).trim(); if (ctx.measureText(t).width <= maxW || !cur) cur = t; else { lines.push(cur); cur = word; } }
    return cur ? lines.concat([cur]) : lines;
  }

  function renderLabel(veg, w, h, ch, edition, dateStr) {
    const mm = v => Math.round(v / 25.4 * CFG.label.dpi), W = mm(CFG.label.mm[0]), H = mm(CFG.label.mm[1]);
    const c = canvas(W, H), x = c.getContext("2d"), GREEN = "rgb(34,85,51)", INK = "rgb(40,30,30)", GREY = "rgb(95,95,95)";
    x.fillStyle = "rgb(250,244,228)"; x.fillRect(0, 0, W, H);
    x.strokeStyle = GREEN; x.lineWidth = 5; x.beginPath(); x.roundRect(14.5, 14.5, W - 29, H - 29, 36); x.stroke();
    // spaced header
    x.font = `800 28px ${FONT.display}`; x.fillStyle = GREEN; x.textBaseline = "middle";
    const head = "ONE OF A KIND", gap = 5, widths = [...head].map(ch2 => x.measureText(ch2).width);
    let hx = W / 2 - (widths.reduce((a, b) => a + b, 0) + gap * (head.length - 1)) / 2;
    [...head].forEach((ch2, i) => { x.fillText(ch2, hx, 52); hx += widths[i] + gap; });
    // dressed-up veg
    const comp = trim(compose(veg, w, h, ch).canvas);
    const bl = 36, bt = 84, bw = W - 72, bh = 560 - 84, s = Math.min(bw * 0.96 / comp.width, bh * 0.92 / comp.height);
    const vw = comp.width * s, vh = comp.height * s, vx = bl + (bw - vw) / 2, vy = bt + (bh - vh) / 2 - 4;
    x.save(); x.filter = "blur(9px)"; x.fillStyle = "rgba(60,40,20,0.24)";
    x.beginPath(); x.ellipse(vx + vw / 2, vy + vh, vw * 0.35, 12, 0, 0, 2 * Math.PI); x.fill(); x.restore();
    x.imageSmoothingQuality = "high"; x.drawImage(comp, vx, vy, vw, vh);
    // name + tagline
    x.textAlign = "center"; x.fillStyle = INK;
    const name = ch.object_name.toUpperCase(); fitFont(x, name, FONT.display, 800, 76, W - 76); x.fillText(name, W / 2, 604);
    let ts = 32, lines;
    for (;;) { x.font = `italic ${ts}px ${FONT.italic}`; lines = wrap(x, ch.tagline, W - 90); if (lines.length <= 2 || ts <= 20) break; ts -= 2; }
    x.fillStyle = GREEN; lines.slice(0, 2).forEach((l, i) => x.fillText(l, W / 2, 662 + i * (ts + 6)));
    // footer
    x.strokeStyle = "rgb(200,190,165)"; x.lineWidth = 2; x.beginPath(); x.moveTo(60, H - 92); x.lineTo(W - 60, H - 92); x.stroke();
    const vt = ch.veg_type && ch.veg_type !== CFG.no_data ? [ch.veg_type.replace(/\b\w/g, m2 => m2.toUpperCase())] : [];
    x.font = `21px ${FONT.body}`; x.fillStyle = GREY; x.fillText([...vt, `No. ${String(edition).padStart(4, "0")}`, dateStr].join("  ·  "), W / 2, H - 60);
    return c;
  }

  function editorView(veg, w, h, ch) {
    const {canvas: comp, pad} = compose(veg, w, h, ch), room = Math.floor(0.1 * Math.max(w, h));
    let [l, t, r, b] = alphaBox(comp);
    l = Math.max(0, Math.min(l, pad) - room); t = Math.max(0, Math.min(t, pad) - room);
    r = Math.min(comp.width, Math.max(r, pad + w) + room); b = Math.min(comp.height, Math.max(b, pad + h) + room);
    const view = cropCanvas(comp, [l, t, r, b]), cw = view.width, chh = view.height;
    return {image: shrink(view, 900).toDataURL("image/png"), frame: {x0: (pad - l) / cw, y0: (pad - t) / chh, bw: w / cw, bh: h / chh, aspect: cw / chh}};
  }

  function shrink(c, max) { const s = Math.min(1, max / Math.max(c.width, c.height)); if (s >= 1) return c; const o = canvas(c.width * s, c.height * s); o.getContext("2d").drawImage(c, 0, 0, o.width, o.height); return o; }

  // ------------------------------------------------------------------ pipeline
  let CUR = null;   // current item: veg cutout canvas, size, mask, edition, date

  function nextEdition() {
    let n = 1; try { n = (parseInt(localStorage.getItem("oddbunch_edition") || "0", 10) || 0) + 1; localStorage.setItem("oddbunch_edition", String(n)); } catch (e) { /* private mode */ }
    return n;
  }

  async function process(dataUrl, name, offline) {
    const t = {}, T0 = performance.now(); let step = T0;
    const lap = k => { const now = performance.now(); t[k] = (now - step) / 1000; step = now; };
    const img = await loadImage(dataUrl);
    const photo = photoMat(img);
    const out = {ok: false, warnings: [], timings: t, edition: null};
    let seg = null;
    try {
      seg = segment(photo); lap("t_segment_s"); out.warnings = seg.warnings;
      const outline = new cv.Mat(); cv.cvtColor(photo, outline, cv.COLOR_RGB2RGBA);
      const cs = new cv.MatVector(); cs.push_back(seg.contour);
      cv.drawContours(outline, cs, 0, new cv.Scalar(90, 220, 60, 255), Math.max(3, Math.floor(Math.min(photo.rows, photo.cols) / 160))); cs.delete();
      const oc = canvas(outline.cols, outline.rows); cv.imshow(oc, outline); outline.delete();
      out.outline = shrink(oc, 900).toDataURL("image/jpeg", 0.88);
      const f = shapeFeatures(photo, seg); lap("t_features_s");
      const m = await match(photo, seg, f, name, offline); lap("t_match_s");
      const [x, y, w, h] = seg.bbox;
      // veg cutout (box-sized canvas with soft alpha) + mask bytes for limb attachment
      const roi = photo.roi(new cv.Rect(x, y, w, h)), mroi = seg.mask.roi(new cv.Rect(x, y, w, h)), alpha = new cv.Mat();
      cv.GaussianBlur(mroi, alpha, new cv.Size(3, 3), 0);
      const rgba = new cv.Mat(); cv.cvtColor(roi, rgba, cv.COLOR_RGB2RGBA);
      for (let i = 0; i < alpha.data.length; i++) rgba.data[i * 4 + 3] = alpha.data[i];
      const veg = canvas(w, h); cv.imshow(veg, rgba);
      const maskBytes = new Uint8Array(mroi.data); roi.delete(); mroi.delete(); alpha.delete(); rgba.delete();
      let ch = m.character;
      if (m.source === "gemini") ch = attachLimbs(ch, maskBytes, w, h);
      const edition = nextEdition(), date = new Date().toLocaleDateString("en-GB", {day: "2-digit", month: "short", year: "numeric"});
      CUR = {veg, w, h, edition, date};
      const label = renderLabel(veg, w, h, ch, edition, date); lap("t_render_s");
      Object.assign(out, {ok: true, edition, character: ch, match: m, label: label.toDataURL("image/png"), ...(() => { const e = editorView(veg, w, h, ch); return {editor_image: e.image, frame: e.frame}; })()});
    } catch (e) {
      out.error = e.message || String(e);
    } finally {
      t.t_total_s = (performance.now() - T0) / 1000;
      photo.delete(); if (seg) { seg.mask.delete(); seg.contour.delete(); }
    }
    return out;
  }

  function cleanCharacter(c) {
    const S = CFG.safety, out = {};
    for (const k of ["resembles", "object_name", "tagline"]) out[k] = String(c[k] || "").trim().slice(0, 80);
    out.veg_type = String(c.veg_type || CFG.no_data).trim().slice(0, 40);
    const n = (v, lo, hi, d) => isNum(v) ? clamp(v, lo, hi) : d;
    out.props = (c.props || []).slice(0, S.max_props).filter(p => p && CFG.props[p.type]).map(p => ({
      type: p.type, x: r3(n(p.x, ...S.xy_range, 0.5)), y: r3(n(p.y, ...S.xy_range, 0.5)),
      size: r3(n(p.size, ...S.size_range, 0.12)), angle: Math.round(n(p.angle, -180, 180, 0) * 10) / 10, flip: !!p.flip}));
    return out;
  }

  function rerender(character) {
    const ch = cleanCharacter(character), [, problems] = validate(ch);
    const label = renderLabel(CUR.veg, CUR.w, CUR.h, ch, CUR.edition, CUR.date), e = editorView(CUR.veg, CUR.w, CUR.h, ch);
    return {ok: true, character: ch, warnings: problems, label: label.toDataURL("image/png"), editor_image: e.image, frame: e.frame};
  }

  return {init, process, rerender, checkKey, getKey, KEY_STORE};
})();
