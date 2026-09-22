/* A finite categorical demonstration. The learning rules live in core.js. */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) =>
    String(s).replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
  const mathOptions = {
    throwOnError: true,
    strict: "error",
    trust: false,
    output: "htmlAndMathml",
  };
  const tex = (source, displayMode = false) =>
    katex.renderToString(source, { ...mathOptions, displayMode });
  function typeset(element) {
    renderMathInElement(element, {
      ...mathOptions,
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
      ],
      errorCallback: (_message, error) => {
        throw error;
      },
    });
  }
  const names = LabContent.names,
    order = Object.keys(RhoAlgorithms);
  const chapters = ["basics", ...order];
  const chapterName = (id) => (id === "basics" ? "Start here" : names[id]);
  const defaultMethods = order.slice();
  const featuredPresets = ["bell", "uniform", "right", "binary"];
  const defaults = () => ({
    ...LabConfig.defaults(),
    preset: "bell",
    optimizer: "adam",
    lr: 0.05,
    evalK: 4,
  });
  let cfg = defaults(),
    selected = defaultMethods.slice(),
    focus = "grpo";
  // Playback is fixed at 8×: one update every 150 ms.
  const pace = 150,
    horizons = [25, 100, 300, 1000];
  let horizon = 1000,
    sim,
    shown = 0;
  let playing = false,
    timer = 0,
    scheduled = null,
    hoverStep = null,
    guide = "basics";
  document.documentElement.style.setProperty("--update-duration", `${pace}ms`);
  const percent = (x) =>
    x > 0 && x < 0.001 ? "<0.1%" : (100 * x).toFixed(1) + "%";
  const signed = (x, digits = 3) => {
    const rounded = Number(x.toFixed(digits));
    return (rounded > 0 ? "+" : "") + rounded.toFixed(digits);
  };
  const setStatus = (text) => {
    $("status").textContent = text;
  };
  const options = (el, entries) => {
    el.innerHTML = Object.entries(entries)
      .map(([v, t]) => `<option value="${esc(v)}">${esc(t)}</option>`)
      .join("");
  };
  options(
    $("preset"),
    Object.fromEntries(featuredPresets.map((id) => [id, RewardLab.presets[id]])),
  );
  options($("transform"), LabContent.transforms);
  options($("judge"), LabContent.judges);
  options(
    $("algorithm-select"),
    Object.fromEntries(chapters.map((id) => [id, chapterName(id)])),
  );
  $("method-options").innerHTML = order
    .map(
      (id) =>
        `<label><input type="checkbox" name="method" value="${id}">${names[id]}</label>`,
    )
    .join("");
  $("algorithm-links").innerHTML = chapters
    .map(
      (id) =>
        `<a href="#algorithms/${id}" data-algorithm="${id}">${chapterName(id)}</a>`,
    )
    .join("");

  function validateSetup(data) {
    if (!data || typeof data !== "object" || Array.isArray(data))
      throw Error("A setup must be an object.");
    const given = data.cfg ?? {};
    if (typeof given !== "object" || Array.isArray(given))
      throw Error("A setup must be a JSON object.");
    // Every run uses Adam. Links made before that carry an SGD-scale learning
    // rate with no optimizer field; those fall back to the Adam default.
    const legacy =
      Object.hasOwn(given, "lr") && !Object.hasOwn(given, "optimizer");
    const config = LabConfig.validate({
      ...defaults(),
      ...given,
      ...(legacy ? { lr: defaults().lr } : {}),
      optimizer: "adam",
    });
    if (
      config.model !== "independent" ||
      config.layout !== "quality" ||
      config.unit
    )
      throw Error(
        "This visualizer uses independent categorical policies and original reward positions.",
      );
    if (config.k > config.n)
      throw Error("Best-of-k must be at most the samples per update.");
    const methods = data.methods || defaultMethods;
    if (
      !Array.isArray(methods) ||
      !methods.length ||
      methods.length > order.length ||
      methods.some((id) => !order.includes(id)) ||
      new Set(methods).size !== methods.length
    )
      throw Error("Choose at least one valid algorithm, with no duplicates.");
    if (!horizons.includes(data.horizon ?? 1000))
      throw Error("Invalid run length.");
    return {
      cfg: config,
      methods,
      horizon: data.horizon ?? 1000,
      focus: methods.includes(data.focus) ? data.focus : methods[0],
    };
  }
  let startupError = "";
  try {
    const param = new URL(location.href).searchParams.get("setup");
    if (param) {
      if (param.length > 60000) throw Error("Setup link is too long.");
      const valid = validateSetup(JSON.parse(param));
      ({ cfg, horizon, focus } = valid);
      selected = valid.methods;
    }
  } catch (error) {
    startupError = `Could not load this setup: ${error.message} Showing defaults.`;
  }

  const numericFields = [
    "n",
    "lr",
    "seed",
    "k",
    "tailAt",
    "bulkScale",
    "tailScale",
    "jump",
    "lambda",
    "ppoClip",
    "ppoEpochs",
    "floor",
  ];
  function writeForm() {
    // Chapter experiments can open presets outside the featured four.
    if (![...$("preset").options].some((o) => o.value === cfg.preset))
      $("preset").add(new Option(RewardLab.presets[cfg.preset], cfg.preset));
    $("preset").value = cfg.preset;
    for (const key of [...numericFields, "transform", "judge"]) {
      const el = $(key);
      if (
        el.tagName === "SELECT" &&
        ![...el.options].some((o) => o.value === String(cfg[key]))
      )
        el.add(new Option(String(cfg[key]), String(cfg[key])));
      el.value = cfg[key];
    }
    $("horizon").value = horizon;
    document.querySelectorAll('[name="method"]').forEach((el) => {
      el.checked = selected.includes(el.value);
    });
    $("settings-error").hidden = true;
    previewMap();
  }
  function previewMap() {
    const type = $("transform").value;
    $("piecewise-settings").hidden = type !== "hinge";
    $("power-settings").hidden = !["boxcox", "yeojohnson"].includes(type);
    const xs = Array.from({ length: 21 }, (_, i) => i / 20);
    const shape = Object.fromEntries(
      ["tailAt", "bulkScale", "tailScale", "jump"].map((k) => [
        k,
        Number($(k).value),
      ]),
    );
    const values = RewardLab.transform(
      xs,
      type,
      Number($("lambda").value),
      sim?.frozen,
      shape,
    );
    if (values.some((v) => !Number.isFinite(v))) return;
    const lo = Math.min(0, ...values),
      hi = Math.max(...values, lo + 1e-6);
    const x = (r) => 30 + r * 160,
      y = (r) => 108 - ((r - lo) / (hi - lo)) * 85;
    const dependent = ["zscore", "ranknormal"].includes(type);
    const points = xs.map((r, i) => `${x(r)},${y(values[i])}`).join(" ");
    $("map-preview").innerHTML =
      `<svg viewBox="0 0 210 142" role="img" aria-label="Training score as a function of judge score"><path d="M30 15V108H195" fill="none" stroke="var(--axis)"/><text x="2" y="25">${hi.toFixed(1)}</text><text x="2" y="110">${lo.toFixed(1)}</text><text x="28" y="123">0</text><text x="187" y="123">1</text><text x="85" y="140">Judge score →</text>${dependent || (type === "hinge" && shape.jump > 0) ? "" : `<polyline points="${points}" fill="none" stroke="var(--green)" stroke-width="1.5"/>`}${xs.map((r, i) => `<circle cx="${x(r)}" cy="${y(values[i])}" r="2" fill="var(--green)"/>`).join("")}</svg>`;
    $("map-note").textContent =
      `Training score ↑. ${LabContent.transformNotes[type]} ${dependent ? "Preview uses one of each reward; actual scores are refitted to every sampled group. Normal scores do not make the policy Gaussian." : "This maps scores, not probabilities."} MaxRL bypasses this map and thresholds the judge score at 0.9.`;
  }
  function apply(data, keepOpen = false) {
    const valid = validateSetup(data);
    pause();
    ({ cfg, horizon, focus } = valid);
    selected = valid.methods;
    reset();
    writeForm();
    if (!keepOpen) $("advanced").open = false;
    // A copied setup describes the committed controls, never a half-edited form.
    history.replaceState(null, "", location.pathname + location.hash);
  }
  function readForm() {
    const input = { ...cfg };
    for (const key of numericFields) {
      if (!$(key).value.trim()) throw Error(`${key} needs a value.`);
      input[key] = Number($(key).value);
    }
    for (const key of ["transform", "judge"]) input[key] = $(key).value;
    input.preset = $("preset").value;
    input.evalK = input.k;
    return input;
  }
  // Simulation settings restart the run, so they commit as soon as a field
  // changes; nothing waits for the panel to close.
  function commitForm(keepOpen) {
    try {
      apply(
        {
          cfg: readForm(),
          methods: [
            ...document.querySelectorAll('[name="method"]:checked'),
          ].map((el) => el.value),
          focus,
          horizon: Number($("horizon").value),
        },
        keepOpen,
      );
    } catch (error) {
      $("settings-error").textContent = error.message;
      $("settings-error").hidden = false;
      $("settings-error").scrollIntoView({ block: "nearest" });
    }
  }
  $("settings").addEventListener("submit", (event) => {
    event.preventDefault();
    commitForm(false);
  });
  $("settings").addEventListener("change", (event) => {
    // Algorithms and run length have their own handlers and keep the run.
    if (event.target.matches('#horizon, [name="method"]')) return;
    commitForm(true);
  });
  $("settings").addEventListener("input", previewMap);
  // Every algorithm is simulated on each update, so the comparison can change
  // without restarting the run.
  $("method-options").addEventListener("change", (event) => {
    const checked = [
      ...document.querySelectorAll('[name="method"]:checked'),
    ].map((el) => el.value);
    if (!checked.length) {
      event.target.checked = true;
      return;
    }
    selected = checked;
    if (!selected.includes(focus)) focus = selected[0];
    buildCards();
    render();
    history.replaceState(null, "", location.pathname + location.hash);
  });
  // Run length only decides where playback stops, so it can change mid-run.
  // Shortening below the recorded updates would discard history; that needs
  // a restart.
  $("horizon").onchange = () => {
    const next = Number($("horizon").value);
    if (next < sim.step) {
      $("settings-error").textContent =
        `This run already has ${sim.step} recorded updates. Choose at least ${sim.step}, or press Restart run to start a shorter one.`;
      $("settings-error").hidden = false;
      return;
    }
    $("settings-error").hidden = true;
    horizon = next;
    render();
    history.replaceState(null, "", location.pathname + location.hash);
  };
  $("close-advanced").onclick = () => {
    writeForm();
    $("advanced").open = false;
    $("advanced").querySelector("summary").focus();
  };
  $("advanced").addEventListener("toggle", () => {
    if ($("advanced").open) pause();
    else writeForm();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && $("advanced").open) $("close-advanced").click();
  });
  $("defaults").onclick = () =>
    apply({ cfg: defaults(), methods: defaultMethods });
  $("preset").onchange = () =>
    apply({
      cfg: { ...cfg, preset: $("preset").value },
      methods: selected,
      horizon,
      focus,
    });


  const plotX = (i) => 38 + i * 14,
    plotY = (p) => 202 - 170 * p;
  function plotMarkup(id) {
    return `<svg class="reward-plot" viewBox="0 0 338 240" role="img" aria-label="${names[id]} outcome probabilities, zero to one hundred percent"><text x="4" y="17">Probability</text>${[0, 0.5, 1].map((v) => `<path class="gridline" d="M32 ${plotY(v)}H328"/><text x="2" y="${plotY(v) + 3}">${100 * v}%</text>`).join("")}${sim.base.map((p, i) => `<rect class="initial-bar" x="${plotX(i) - 6}" y="${plotY(p)}" width="12" height="${p * 170}"/><rect class="bar" data-bin="${i}" x="${plotX(i) - 4}" y="${plotY(p)}" width="8" height="${p * 170}"><title></title></rect>`).join("")}<path class="axisline" d="M32 202H328"/>${[0, 10, 20].map((i) => `<text text-anchor="middle" x="${plotX(i)}" y="218">${i / 20}</text>`).join("")}<text text-anchor="middle" x="180" y="237">Original reward →</text></svg>`;
  }
  function buildCards() {
    $("comparison").dataset.count = selected.length;
    $("comparison").innerHTML = selected
      .map(
        (id) =>
          `<article class="algorithm-card" data-method="${id}" style="--series:var(--${id})"><div class="card-heading"><h2>${names[id]}</h2><a href="#algorithms/${id}">Read ↗</a></div><p class="card-tag">${RhoAlgorithms[id].short}</p>${plotMarkup(id)}<div class="card-metrics">${[
            [
              "mean",
              "Mean reward",
              "Exact expected original reward under the displayed policy.",
            ],
            [
              "best",
              `Expected best of ${cfg.evalK}`,
              `Exact expected maximum original reward of ${cfg.evalK} independent draws, assuming an oracle selects the best.`,
            ],
            [
              "entropy",
              "Entropy",
              "Entropy of the displayed policy in nats; ln 21 ≈ 3.04 is uniform, 0 is a single outcome.",
            ],
          ]
            .map(
              ([key, label, definition]) =>
                `<div title="${esc(definition)}"><span>${label}</span><b data-metric="${key}"></b><small data-change="${key}"></small></div>`,
            )
            .join("")}</div></article>`,
      )
      .join("");
  }
  // The step charts share one cursor. Hovering either one moves it, the
  // readouts and the best-of-k chart to that update; leaving returns them to
  // the displayed update.
  const stepCharts = { mean: null, entropy: null };
  const cursorStep = () =>
    hoverStep === null ? shown : Math.min(hoverStep, shown);
  function renderReadouts(step) {
    const frame = sim.history[step];
    const row = (format) =>
      selected
        .map(
          (id) =>
            `<span style="--series:var(--${id})"><i></i>${names[id]} <b>${format(frame.methods[id].metrics)}</b></span>`,
        )
        .join("");
    $("values-mean").innerHTML = row((m) => m.mean.toFixed(3));
    $("values-entropy").innerHTML = row((m) => m.entropy.toFixed(2));
    $("values-best").innerHTML = row((m) => `k=${cfg.evalK}: ${m.best.toFixed(3)}`);
  }
  function renderBest(step) {
    const plot = RhoCurves.best(sim.history, selected, step, sim.rewards);
    $("curve-best").innerHTML = RhoCurves.bestSvg(plot, names);
    $("curve-best-title").textContent = `Expected best of k draws · update ${step}`;
  }
  function moveCursors(step) {
    for (const metric of Object.keys(stepCharts))
      RhoCurves.moveCursor($(`curve-${metric}`), stepCharts[metric], step);
    renderBest(step);
    renderReadouts(step);
  }
  function renderCurves() {
    const step = cursorStep();
    for (const metric of Object.keys(stepCharts)) {
      stepCharts[metric] = RhoCurves.data(sim.history, selected, metric, shown, horizon);
      $(`curve-${metric}`).innerHTML = RhoCurves.svg(stepCharts[metric], names, step);
    }
    renderBest(step);
    renderReadouts(step);
  }
  for (const metric of Object.keys(stepCharts)) {
    const container = $(`curve-${metric}`);
    container.addEventListener("pointermove", (event) => {
      const box = container.querySelector("svg")?.getBoundingClientRect();
      if (!box || !box.width) return;
      const step = RhoCurves.stepAt(
        stepCharts[metric],
        (event.clientX - box.left) / box.width,
      );
      if (step === hoverStep) return;
      hoverStep = step;
      moveCursors(cursorStep());
    });
    container.addEventListener("pointerleave", () => {
      if (hoverStep === null) return;
      hoverStep = null;
      moveCursors(cursorStep());
    });
  }
  function render() {
    const frame = sim.history[shown];
    for (const card of document.querySelectorAll(".algorithm-card")) {
      const id = card.dataset.method,
        state = frame.methods[id],
        start = sim.history[0].methods[id].metrics;
      card.querySelectorAll(".bar").forEach((bar, i) => {
        const p = state.p[i];
        bar.setAttribute("y", plotY(p));
        bar.setAttribute("height", p * 170);
        bar.querySelector("title").textContent =
          `Reward ${(i / 20).toFixed(2)}: ${percent(p)}; start ${percent(sim.base[i])}`;
      });
      for (const key of ["mean", "best", "entropy"]) {
        const m = state.metrics[key],
          delta = m - start[key],
          digits = key === "entropy" ? 2 : 3;
        card.querySelector(`[data-metric="${key}"]`).textContent = m.toFixed(digits);
        card.querySelector(`[data-change="${key}"]`).textContent =
          shown === 0 ? "at start" : `${signed(delta, digits)} from start`;
      }
    }
    $("play").textContent = playing
      ? "Ⅱ Pause"
      : shown < sim.step
        ? "▶ Replay"
        : sim.step >= horizon
          ? "↻ Replay"
          : "▶ Play";
    $("previous").disabled = shown === 0;
    $("step").disabled = shown >= horizon && shown === sim.step;
    renderCurves();
  }
  function schedule(callback, delay) {
    clearTimeout(timer);
    scheduled = { callback, at: performance.now() + delay };
    timer = setTimeout(() => {
      scheduled = null;
      callback();
    }, delay);
  }
  function pause() {
    clearTimeout(timer);
    scheduled = null;
    playing = false;
    if (sim) render();
  }
  // Replays a recorded update if there is one, otherwise computes the next.
  function advance() {
    if (!playing) return;
    const started = performance.now();
    if (shown < sim.step) shown++;
    else if (sim.step < horizon) {
      RewardLab.step(sim);
      shown = sim.step;
    }
    render();
    if (shown >= horizon) pause();
    // Keep the cadence at one update per pace, net of compute and drawing.
    else schedule(advance, Math.max(0, pace - (performance.now() - started)));
  }
  $("play").onclick = () => {
    if (playing) {
      pause();
      return;
    }
    if (shown === horizon) shown = 0;
    playing = true;
    advance();
  };
  $("step").onclick = () => {
    pause();
    if (shown < sim.step) shown++;
    else if (sim.step < horizon) {
      RewardLab.step(sim);
      shown = sim.step;
    }
    render();
  };
  $("reset").onclick = () => reset();
  function replayTo(requested) {
    pause();
    shown = Math.max(0, Math.min(requested, sim.step));
    render();
  }
  $("previous").onclick = () => replayTo(shown - 1);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) pause();
  });
  function reset() {
    pause();
    // Only the methods this page can show are simulated.
    sim = RewardLab.create({ ...cfg, methods: order });
    shown = 0;
    $("task-n").textContent = cfg.n.toLocaleString("en-US");
    buildCards();
    render();
    setStatus("");
  }

  function clipWidget() {
    return `<h3>What clipping does</h3><p>For one sampled outcome, compare the plain ratio term with the clipped surrogate. The advantage stays fixed during these epochs. Clipping acts on this objective, not on the reward.</p><div class="clip-controls"><label>Advantage<select id="clip-sign"><option value="1">Positive · A = +1</option><option value="-1">Negative · A = −1</option></select></label><label><span>Probability ratio \\(q\\): <output id="ratio-value">1.40</output></span><input id="clip-ratio" type="range" min="0" max="2" step="0.01" value="1.4"></label><label><span>Clip width \\(\\epsilon\\): <output id="epsilon-value">0.20</output></span><input id="clip-epsilon" type="range" min="0.05" max="0.5" step="0.01" value="0.2"></label></div><svg id="clip-figure" viewBox="0 0 640 225" role="img" aria-label="Plain ratio term and clipped surrogate for one sample"></svg><p id="clip-reading"></p>`;
  }
  function renderClip() {
    const a = Number($("clip-sign").value),
      q = Number($("clip-ratio").value),
      e = Number($("clip-epsilon").value);
    const value = (r) =>
      Math.min(r * a, Math.max(1 - e, Math.min(1 + e, r)) * a);
    const x = (r) => 50 + 270 * r,
      y = (v) => (a > 0 ? 182 - 70 * v : 42 - 70 * v);
    const clipped = a > 0 ? q > 1 + e : q < 1 - e;
    const boundary = Math.abs(q - (a > 0 ? 1 + e : 1 - e)) < 1e-9;
    $("ratio-value").textContent = q.toFixed(2);
    $("epsilon-value").textContent = e.toFixed(2);
    $("clip-figure").innerHTML =
      `<rect x="${x(1 - e)}" y="26" width="${540 * e}" height="163" fill="var(--soft)"/><path d="M50 ${y(0)}H600" stroke="var(--axis)"/>${[0, 1, 2].map((r) => `<text x="${x(r)}" y="205" text-anchor="middle" fill="var(--muted)" font-size="12">${r}</text>`).join("")}<text x="330" y="222" text-anchor="middle" fill="var(--muted)" font-size="11">Probability ratio q</text><text x="18" y="${y(0) + 4}" fill="var(--muted)" font-size="11">0</text><text x="14" y="${y(2 * a) + 4}" fill="var(--muted)" font-size="11">${2 * a}</text><path d="M${x(0)} ${y(0)}L${x(2)} ${y(2 * a)}" stroke="var(--axis)" stroke-width="2" stroke-dasharray="5 4"/><polyline points="${Array.from({ length: 201 }, (_, i) => `${x(i / 100)},${y(value(i / 100))}`).join(" ")}" fill="none" stroke="var(--green)" stroke-width="3"/><circle cx="${x(q)}" cy="${y(value(q))}" r="5" fill="var(--green)"/><text x="425" y="18" font-size="11" fill="var(--muted)">Dashed: qA · Solid: clipped</text>`;
    $("clip-reading").innerHTML =
      `At ${tex("q = " + q.toFixed(2))}, the surrogate is ${value(q).toFixed(2)}. ${boundary ? "This is a corner of the surrogate; its slope changes here." : clipped ? "The slope is zero for this sample: pushing farther in this direction adds no objective gain." : `This sample still has slope ${a > 0 ? "+1" : "−1"}.`} The flat side switches for negative advantages. Other samples can still move this probability.`;
  }
  function bindWorked() {
    let stage = 0;
    const n = 3,
      baseline = 0.4,
      k = 2;
    const settings = {
      lr: 0.5,
      unit: false,
      ppoEpochs: 4,
      ppoClip: 0.2,
    };
    const groups = {
      mixed: [0.2, 0.5, 1],
      binary: [0, 0, 1],
      tied: [0.5, 0.5, 0.5],
      success: [1, 1, 1],
    };
    const fmt = (v) =>
      Math.abs(v) < 0.0005 ? "0" : String(Number(v.toFixed(3)));
    const inline = (source) => tex(source);
    function draw() {
      const raw = groups[$("lesson-group").value],
        map = $("lesson-map").value;
      const scores =
        guide === "maxrl"
          ? raw.map((r) => Number(r >= 0.9))
          : RewardLab.transform(raw, map);
      const weights = RewardLab.advantage(scores, guide, k, baseline);
      // Condition on one draw of each supported answer; use the existing optimizers.
      const model = RewardLab.create({
        preset: "custom",
        custom: "0.2,0.5,1",
        model: "independent",
      }).models[guide];
      const ids = [4, 10, 20],
        before = RewardLab.forward(model);
      const grad = before.map((p) => -p * RewardLab.mean(weights));
      ids.forEach((id, i) => (grad[id] += weights[i] / n));
      const diagnostics =
        guide === "grpo"
          ? RewardLab.ppoUpdate(model, before, ids, weights, settings)
          : { norm: RewardLab.update(model, grad, settings.lr) };
      const after = RewardLab.forward(model),
        mean = RewardLab.mean(scores);
      const sigma = Math.sqrt(
        RewardLab.mean(scores.map((v) => (v - mean) ** 2)),
      );
      $("lesson-settings").textContent = `Independent logits · plain SGD step, learning rate 0.5${guide === "grpo" ? " · 4 epochs · clip width 0.2" : ""}${guide === "pkpo" ? " · k = 2" : ""}.`;
      $("lesson-stage").textContent =
        `${stage + 1} / 3 · ${["Score the answers", "Compute the weights", "Update the probabilities"][stage]}`;
      $("lesson-score-heading").textContent =
        stage === 0
          ? "Score"
          : guide === "maxrl"
            ? "Success"
            : "Training score";
      $("lesson-rows").innerHTML = ids
        .map((id, i) => {
          const p = stage === 2 ? after[id] : before[id],
            delta = p - before[id];
          return `<tr><th scope="row">${["A", "B", "C"][i]}</th><td class="lesson-score">${fmt(stage === 0 ? raw[i] : scores[i])}</td><td class="lesson-weight">${stage > 0 ? fmt(weights[i]) : "—"}</td><td><span class="lesson-probability">${(100 * p).toFixed(1)}%</span><span class="probability-track" aria-hidden="true"><span style="width:${100 * p}%"></span></span>${stage === 2 ? `<small class="probability-change">${Math.abs(delta) < 0.0005 ? "unchanged" : `${delta > 0 ? "+" : "−"}${(100 * Math.abs(delta)).toFixed(1)} pp`}</small>` : ""}</td></tr>`;
        })
        .join("");
      let explanation;
      if (stage === 0) {
        explanation =
          "These are three distinct answers, with scores assigned by a checker. No learning has happened yet. Each answer still has a one-third chance. Next, the selected method turns the training scores into weights.";
        if (guide === "maxrl")
          explanation +=
            " MaxRL first turns these scores into pass/fail labels at 0.9; the selected continuous reward map is bypassed.";
        else if (map !== "identity")
          explanation += ` The ${map === "square" ? "square" : "2r + 1"} map will change the training scores before weights are computed.`;
      } else if (stage === 1) {
        const last = fmt(scores[2]);
        switch (guide) {
          case "rloo":
            explanation = `For C, the other two scores average ${inline(String.raw`(${fmt(scores[0])}+${fmt(scores[1])})/2=${fmt((scores[0] + scores[1]) / 2)}`)}. Subtract that baseline: ${inline(String.raw`A_C=${last}-${fmt((scores[0] + scores[1]) / 2)}=${fmt(weights[2])}`)}. Repeat while leaving out A, then B.`;
            break;
          case "grpo":
            explanation =
              sigma === 0
                ? "All scores agree. The standard deviation is zero, so this implementation explicitly sets every advantage to zero. No division by zero is performed."
                : `The mean is ${inline(String.raw`\mu=${fmt(mean)}`)} and the standard deviation is ${inline(String.raw`\sigma=${fmt(sigma)}`)}. For C, ${inline(String.raw`A_C=\frac{${last}-${fmt(mean)}}{${fmt(sigma)}}\approx ${fmt(weights[2])}`)}. The small numerical stabilizer is included in the computed table. These advantages stay fixed during the clipped epochs.`;
            break;
          case "maxrl":
            explanation =
              mean === 0
                ? "Every label is a failure. The success rate is zero, so this implementation skips the group and uses zero weights. A continuous score improvement below 0.9 cannot change that."
                : `The success fraction is ${inline(String.raw`\widehat p=${fmt(mean)}`)}. Subtract it from each binary label, then divide by it. C gets ${inline(String.raw`A_C=\frac{${last}-${fmt(mean)}}{${fmt(mean)}}\approx ${fmt(weights[2])}`)}. The reward map is bypassed.${mean === 1 ? " With all successes, every centered weight is zero." : ""}`;
            break;
          case "pkpo":
            explanation = `With k = 2, each partner set contains just one of the other answers. For C, average its nonnegative improvement over A and B, then multiply by 2: ${inline(String.raw`A_C=(${last}-${fmt(scores[0])})_++(${last}-${fmt(scores[1])})_+=${fmt(weights[2])}`)}. Repeat for the other answers. A zero weight means no marginal gain for these partner sets.`;
            break;
          case "tailrl": {
            const sorted = scores.slice().sort((a, b) => a - b),
              first = (sorted[1] - sorted[0]) / 2,
              second = sorted[2] - sorted[1];
            explanation = `For these sorted scores, the first gap is ${fmt(sorted[1] - sorted[0])}, shared by two surviving answers; the next is ${fmt(second)}, with one survivor. Accumulated values are ${inline(`[0,${fmt(first)},${fmt(first + second)}]`)}. Subtract their mean and multiply by 3 to get the table’s weights. Ties add zero gaps. With this mean-gradient convention, three samples represent finite order T = 2.`;
            break;
          }
        }
      } else {
        explanation =
          "The table now shows probabilities after the update. Scores stayed fixed. The new probabilities come from updated logits and softmax; they are not normalized versions of the weight column. “pp” means percentage points of probability.";
        if (guide === "grpo")
          explanation += ` Four clipped epochs reused this group. The peak fraction of samples whose surrogate gradient was clipped to zero was ${(100 * diagnostics.clipped).toFixed(0)}%; the resulting KL(old ∥ new) is ${fmt(diagnostics.kl)}.`;
        else
          explanation += ` All three weight contributions were combined in one gradient step. A positive weight alone does not guarantee a probability increase.`;
        if (Math.max(...scores) === Math.min(...scores))
          explanation +=
            " This balanced group has one of each equally likely answer, so constant weights cancel exactly. Random groups in the main visualizer need not be balanced.";
      }
      $("lesson-explanation").innerHTML = `<p>${explanation}</p>`;
      $("lesson-previous").disabled = stage === 0;
      $("lesson-next").disabled = stage === 2;
    }
    $("lesson-previous").onclick = () => {
      stage = Math.max(0, stage - 1);
      draw();
    };
    $("lesson-next").onclick = () => {
      stage = Math.min(2, stage + 1);
      draw();
    };
    for (const id of ["lesson-group", "lesson-map"])
      $(id).onchange = () => {
        stage = 0;
        draw();
      };
    draw();
  }

  function renderGuide() {
    $("algorithm-select").value = guide;
    document
      .querySelectorAll("[data-algorithm]")
      .forEach((el) =>
        el.setAttribute("aria-current", String(el.dataset.algorithm === guide)),
      );
    const pos = chapters.indexOf(guide),
      prev = chapters[pos - 1],
      next = chapters[pos + 1];
    const readingNav = `<nav class="reading-nav" aria-label="Continue reading">${prev ? `<a href="#algorithms/${prev}">← ${chapterName(prev)}</a>` : "<span></span>"}${next ? `<a href="#algorithms/${next}">Next: ${chapterName(next)} →</a>` : `<a href="#visualizer">Open visualizer →</a>`}</nav>`;
    if (guide === "basics") {
      $("algorithm-article").innerHTML =
        $("basics-template").innerHTML + readingNav;
      typeset($("algorithm-article"));
      return;
    }
    const data = RhoAlgorithms[guide],
      source = LabContent.methods.find((m) => m.id === guide);
    const paragraphs = (list) => list.map((p) => `<p>${esc(p)}</p>`).join("");
    const clippedUpdate =
      guide === "grpo"
        ? `<small>These advantages enter the clipped update below. \\(q_i\\) is current probability / rollout probability; \\(\\widetilde{q}_i\\) clips it to \\([1-\\epsilon,1+\\epsilon]\\). \\(\\ell_i\\) is one sample’s surrogate and \\(n\\) is group size.</small>${tex(RhoAlgorithms.grpo.clippedEquation, true)}`
        : "";
    $("algorithm-article").innerHTML =
      `<h2>${names[guide]}</h2><p class="full-name">${esc(data.name)}</p><p class="guide-lead">${esc(data.lead)}</p>${paragraphs(data.why)}
      <p class="basics-link"><a href="#algorithms/basics">Policy and gradient basics</a></p>
      <h3>Update rule</h3>${paragraphs(data.read)}
      <div class="equation">${tex(data.equation, true)}${clippedUpdate}${data.equationNote ? `<small>${esc(data.equationNote)}</small>` : ""}</div>
      <details class="lesson-detail"><summary>Symbols</summary><dl class="terms">${data.terms.map(([k, v]) => `<dt>${tex(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl></details>
      <a class="source-link" href="${esc(source.url)}" target="_blank" rel="noreferrer">${esc(data.citation)} ↗</a>
      ${$("worked-template").innerHTML}
      ${data.clipped ? clipWidget() : ""}
      <h3>Changing the reward scale</h3>${paragraphs(data.shaping)}
      <h3>Application to language models</h3><p>${esc(data.transfer)}</p>
      <details class="lesson-detail"><summary>What this implementation includes</summary><p>${esc(data.here)}</p></details>
      <h3>Limitations and experiments</h3><p class="hint">These experiments test the finite setup. They are not reported LLM results from the papers.</p>
      <div class="lesson-cases">${data.cases.map((c, i) => `<section><h4>${esc(c.title)}</h4><p>${esc(c.text)}</p><button ${i === 0 ? 'id="try-example"' : ""} data-case="${i}">Open experiment →</button></section>`).join("")}</div>
      ${readingNav}`;
    typeset($("algorithm-article"));
    bindWorked();
    document.querySelectorAll("[data-case]").forEach(
      (button) =>
        (button.onclick = () => {
          const chosen = data.cases[Number(button.dataset.case)];
          apply({
            cfg: {
              ...defaults(),
              ...chosen.settings,
              evalK: chosen.settings.k || 4,
            },
            methods: data.comparison,
            focus: guide,
            horizon,
          });
          location.hash = "visualizer";
          $("play").focus();
        }),
    );
    if (data.clipped) {
      for (const id of ["clip-sign", "clip-ratio", "clip-epsilon"])
        $(id).oninput = renderClip;
      renderClip();
    }
  }
  function route(event) {
    if (location.hash === "#main") {
      $("main").focus();
      return;
    }
    const parts = location.hash.slice(1).split("/");
    const isGuide = parts[0] === "algorithms";
    if (isGuide) {
      pause();
      guide = chapters.includes(parts[1]) ? parts[1] : guide;
      renderGuide();
    }
    $("visualizer").hidden = isGuide;
    $("algorithms").hidden = !isGuide;
    $("nav-visualizer").toggleAttribute("aria-current", !isGuide);
    $("nav-algorithms").toggleAttribute("aria-current", isGuide);
    (isGuide ? $("nav-algorithms") : $("nav-visualizer")).setAttribute(
      "aria-current",
      "page",
    );
    document.title = isGuide
      ? `${chapterName(guide)} | ρ`
      : "ρ | RL visualizer";
    $("advanced").open = false;
    if (event) {
      window.scrollTo(0, 0);
      $("main").focus({ preventScroll: true });
    }
  }
  $("algorithm-select").onchange = () => {
    location.hash = `algorithms/${$("algorithm-select").value}`;
  };
  window.addEventListener("hashchange", route);
  typeset($("settings"));
  reset();
  writeForm();
  route();
  if (startupError) setStatus(startupError);
  // Expose experiment state for reproducibility checks and small extensions.
  window.Rho = {
    get state() {
      return {
        cfg: { ...cfg },
        methods: selected.slice(),
        focus,
        shown,
        playing,
        horizon,
        sim,
      };
    },
  };
})();
