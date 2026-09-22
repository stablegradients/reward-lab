"use strict";
/* Learning-curve charts. Step charts share one cursor so a hover on either
   moves the readouts and the best-of-k chart to that update. Their x-axis
   grows with the recorded updates, so early progress is readable. */
const RhoCurves = (() => {
  const metrics = { mean: "Mean original reward", entropy: "Entropy (nats)" };
  // One coordinate system for every chart: a 480 × 300 box with the plot
  // area from (48, 24) to (468, 250).
  const left = 48,
    right = 468,
    top = 24,
    bottom = 250,
    ks = Array.from({ length: 64 }, (_, i) => i + 1);
  // Reward axes zoom in: they run from 90% of the smallest plotted value up
  // to 1, so a run that starts near 0.5 fills the chart instead of the top
  // half. Entropy keeps its full range.
  const zoomed = (values) => ({
    lo: Math.max(0, 0.9 * Math.min(...values)),
    hi: 1,
  });
  const y = (value, range) =>
    bottom - ((value - range.lo) / (range.hi - range.lo)) * (bottom - top);
  const axis = (range, digits) =>
    [0, 0.5, 1]
      .map((t) => range.lo + t * (range.hi - range.lo))
      .map(
        (v) =>
          `<path d="M${left} ${y(v, range)}H${right}" class="gridline"/><text x="${left - 6}" y="${y(v, range) + 4}" text-anchor="end">${v.toFixed(digits)}</text>`,
      )
      .join("");
  const stepX = (step, span) => left + (step / span) * (right - left);
  const kX = (k) => left + (Math.log2(k) / Math.log2(64)) * (right - left);
  // Labels near either edge anchor away from it so they stay inside the box.
  const anchorAt = (cx) =>
    cx > right - 40 ? "end" : cx < left + 40 ? "start" : "middle";
  function data(history, methods, metric, shown, horizon) {
    if (!Object.hasOwn(metrics, metric))
      throw Error("Unknown learning-curve metric.");
    const series = methods.map((id) => ({
      id,
      points: history
        .slice(0, shown + 1)
        .map((frame, step) => ({ step, value: frame.methods[id].metrics[metric] })),
    }));
    const range =
      metric === "entropy"
        ? { lo: 0, hi: Math.log(history[0].methods[methods[0]].p.length) }
        : zoomed(series.flatMap((s) => s.points.map((p) => p.value)));
    return {
      metric,
      label: metrics[metric],
      range,
      horizon,
      shown,
      span: Math.max(shown, 10),
      series,
    };
  }
  // Converts a pointer position on the rendered chart to a training update.
  function stepAt(plot, fractionAcross) {
    const sx = fractionAcross * 480;
    return Math.max(
      0,
      Math.min(
        plot.shown,
        Math.round(((sx - left) / (right - left)) * plot.span),
      ),
    );
  }
  function svg(plot, names, cursor = plot.shown) {
    const x = (step) => stepX(step, plot.span);
    const at = Math.max(0, Math.min(cursor, plot.shown));
    return `<svg viewBox="0 0 480 300" role="img" aria-label="${plot.label} over training updates"><title>${plot.label} over training updates</title><desc>Selected algorithms through update ${plot.shown}. Values use original rewards, not transformed training scores.</desc>${axis(plot.range, 2)}${[0, 0.5, 1]
      .map(
        (t) =>
          `<text x="${x(Math.round(t * plot.span))}" y="${bottom + 20}" text-anchor="${anchorAt(x(Math.round(t * plot.span)))}">${Math.round(t * plot.span)}</text>`,
      )
      .join("")}<text x="${(left + right) / 2}" y="${bottom + 40}" text-anchor="middle">Training update${plot.shown < plot.horizon ? ` · run continues to ${plot.horizon}` : ""}</text>${plot.series
      .map(
        (s) =>
          `<polyline data-curve="${s.id}" fill="none" stroke="var(--${s.id})" stroke-width="2" points="${s.points.map((p) => `${x(p.step)},${y(p.value, plot.range)}`).join(" ")}"/>`,
      )
      .join("")}<line data-cursor x1="${x(at)}" x2="${x(at)}" y1="${top}" y2="${bottom}" stroke="var(--muted)" stroke-dasharray="4 4"/><text data-cursor-label x="${x(at)}" y="${top - 8}" text-anchor="${anchorAt(x(at))}">update ${at}</text>${plot.series
      .map(
        (s) =>
          `<circle data-marker="${s.id}" cx="${x(at)}" cy="${y(s.points[at].value, plot.range)}" r="4" fill="var(--${s.id})"><title>${names[s.id]}: ${s.points[at].value.toFixed(4)} at update ${at}</title></circle>`,
      )
      .join("")}<rect data-hover x="${left}" y="${top}" width="${right - left}" height="${bottom - top}" fill="transparent"/></svg>`;
  }
  // Moves the cursor of an already rendered step chart without redrawing it.
  function moveCursor(element, plot, cursor) {
    const at = Math.max(0, Math.min(cursor, plot.shown)),
      cx = stepX(at, plot.span);
    const line = element.querySelector("[data-cursor]");
    if (!line) return;
    line.setAttribute("x1", cx);
    line.setAttribute("x2", cx);
    const label = element.querySelector("[data-cursor-label]");
    label.setAttribute("x", cx);
    label.setAttribute("text-anchor", anchorAt(cx));
    label.textContent = `update ${at}`;
    for (const s of plot.series) {
      const marker = element.querySelector(`[data-marker="${s.id}"]`);
      marker.setAttribute("cx", cx);
      marker.setAttribute("cy", y(s.points[at].value, plot.range));
    }
  }
  function best(history, methods, step, rewards) {
    const at = Math.max(0, Math.min(step, history.length - 1));
    // The frame carries best-of-k averaged over prompts; a mixture policy's
    // own best-of-k would overstate it, so only fall back to that for frames
    // without one.
    const series = methods.map((id) => {
      const frame = history[at].methods[id];
      const values =
        frame.bestK ?? RewardLab.bestCurve(frame.p, rewards, ks);
      return { id, points: ks.map((k, i) => ({ k, value: values[i] })) };
    });
    return {
      step: at,
      ks,
      range: zoomed(series.map((s) => s.points[0].value)),
      series,
    };
  }
  function bestSvg(plot, names) {
    return `<svg viewBox="0 0 480 300" role="img" aria-label="Expected best of k draws at update ${plot.step}"><title>Expected best of k draws at update ${plot.step}</title><desc>Exact expected maximum original reward among k independent draws from each policy, assuming an oracle picks the best.</desc>${axis(plot.range, 2)}${[1, 2, 4, 8, 16, 32, 64]
      .map(
        (k) =>
          `<text x="${kX(k)}" y="${bottom + 20}" text-anchor="middle">${k}</text>`,
      )
      .join("")}<text x="${(left + right) / 2}" y="${bottom + 40}" text-anchor="middle">k draws</text>${plot.series
      .map(
        (s) =>
          `<polyline data-curve="${s.id}" fill="none" stroke="var(--${s.id})" stroke-width="2" points="${s.points.map((p) => `${kX(p.k)},${y(p.value, plot.range)}`).join(" ")}"/>${s.points
            .filter((p) => Number.isInteger(Math.log2(p.k)))
            .map(
              (p) =>
                `<circle cx="${kX(p.k)}" cy="${y(p.value, plot.range)}" r="3" fill="var(--${s.id})"><title>${names[s.id]}: ${p.value.toFixed(4)} for k = ${p.k}</title></circle>`,
            )
            .join("")}`,
      )
      .join("")}</svg>`;
  }
  return { metrics, ks, data, stepAt, svg, moveCursor, best, bestSvg };
})();
if (typeof module !== "undefined") module.exports = RhoCurves;
