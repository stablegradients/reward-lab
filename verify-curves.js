"use strict";
const assert = require("node:assert/strict");
global.RewardLab = require("./core.js");
const LabContent = require("./content.js");
const RhoCurves = require("./curves.js");
const run = RewardLab.create({ preset: "bell", seed: 42 });
for (let i = 0; i < 7; i++) RewardLab.step(run);
const before = JSON.stringify(run);
const names = LabContent.names;
for (const metric of Object.keys(RhoCurves.metrics)) {
  const plot = RhoCurves.data(run.history, RewardLab.methods, metric, 3, 100);
  assert.equal(plot.series.length, RewardLab.methods.length);
  for (const s of plot.series) {
    assert.equal(s.points.length, 4); // replay truncation: through the shown update only
    for (const p of s.points) assert(Number.isFinite(p.value) && p.value >= plot.range.lo - 1e-12 && p.value <= plot.range.hi + 1e-12);
  }
  const svg = RhoCurves.svg(plot, names, 2);
  assert(!svg.includes("NaN"));
  assert(svg.includes("update 2") && svg.includes('data-cursor') && svg.includes('data-hover'));
  assert(RhoCurves.svg(plot, names, 99).includes("update 3")); // cursor clamps to the shown update
}
assert.equal(RhoCurves.data(run.history, ["grpo"], "mean", 0, 25).series[0].points.length, 1);
assert.throws(() => RhoCurves.data(run.history, ["grpo"], "high", 3, 100));
// Pointer position → update, clamped to the recorded range.
const plot = RhoCurves.data(run.history, ["grpo"], "mean", 7, 100);
assert.equal(RhoCurves.stepAt(plot, 0), 0);
assert.equal(RhoCurves.stepAt(plot, 1), 7);
assert.equal(RhoCurves.stepAt(plot, 0.1), 0);
assert.equal(plot.span, 10); // the axis spans at least ten updates, then grows with the run
assert.equal(RhoCurves.stepAt(plot, (48 + 420 * 0.5) / 480), 5);
assert.equal(RhoCurves.data(run.history, ["grpo"], "mean", 7, 100).span, 10);
assert.equal(RhoCurves.data(Array(41).fill(run.history[0]), ["grpo"], "mean", 40, 1000).span, 40);
assert(RhoCurves.svg(plot, names).includes("run continues to 100"));
// Best-of-k: exact, monotone in k, matches the single-k metric, and bounded by the top reward.
const best = RhoCurves.best(run.history, RewardLab.methods, 7, run.rewards);
assert.equal(best.step, 7);
assert.equal(best.ks.length, 64);
for (const s of best.series) {
  const values = s.points.map((p) => p.value);
  for (let i = 1; i < values.length; i++) assert(values[i] >= values[i - 1] - 1e-12, "best-of-k must not decrease in k");
  assert(Math.abs(values[run.cfg.evalK - 1] - run.history[7].methods[s.id].metrics.best) < 1e-12, "best-of-k curve must match the single-k metric");
  assert(Math.abs(values[0] - run.history[7].methods[s.id].metrics.mean) < 1e-12 && values.at(-1) <= 1 + 1e-12, "best of one draw is the mean");
}
assert.equal(RhoCurves.best(run.history, ["grpo"], 500, run.rewards).step, 7); // clamps to recorded history
assert.equal(RewardLab.bestKs.length, RhoCurves.ks.length);
// With several answer sets the chart must use the per-prompt average, not the mixture's own best-of-k.
{
  const multi = RewardLab.create({ preset: "bell", seed: 4, contexts: 64, batch: 64, n: 8, optimizer: "adam", lr: 0.5, methods: ["grpo"] });
  for (let i = 0; i < 30; i++) RewardLab.step(multi);
  const frame = multi.history[30].methods.grpo,
    plotted = RhoCurves.best(multi.history, ["grpo"], 30, multi.rewards).series[0].points.map((p) => p.value),
    mixture = RewardLab.bestCurve(frame.p, multi.rewards, RhoCurves.ks);
  assert.deepEqual(plotted, Array.from(frame.bestK));
  assert(Math.abs(plotted[0] - frame.metrics.mean) < 1e-12);
  assert(plotted[3] < mixture[3] - 1e-6, "per-prompt best-of-4 is below the mixture's");
}
// Reward axes zoom: 90% of the smallest plotted value up to 1; entropy keeps 0..ln 21.
const meanPlot = RhoCurves.data(run.history, RewardLab.methods, "mean", 7, 100);
const minMean = Math.min(...meanPlot.series.flatMap((s) => s.points.map((p) => p.value)));
assert(Math.abs(meanPlot.range.lo - 0.9 * minMean) < 1e-12 && meanPlot.range.hi === 1);
assert(Math.abs(best.range.lo - 0.9 * Math.min(...best.series.map((s) => s.points[0].value))) < 1e-12 && best.range.hi === 1);
const entropyPlot = RhoCurves.data(run.history, RewardLab.methods, "entropy", 7, 100);
assert(entropyPlot.range.lo === 0 && Math.abs(entropyPlot.range.hi - Math.log(21)) < 1e-12);
const zero = RewardLab.create({ preset: "zero", seed: 1 }); RewardLab.step(zero);
assert.equal(RhoCurves.data(zero.history, ["grpo"], "mean", 1, 100).range.lo, 0); // never below zero
const svg = RhoCurves.bestSvg(best, names);
assert(!svg.includes("NaN") && svg.includes("update 7") && svg.includes(">64<"));
assert.equal(JSON.stringify(run), before, "charts must not mutate the run");
console.log("PASS: mean and entropy curves, cursor placement and pointer mapping, exact monotone best-of-k curves, and no simulation mutation.");
