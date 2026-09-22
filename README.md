# ρ

An interactive visualizer of reinforcement learning updates on finite reward distributions.

[Visualizer](https://stablegradients.github.io/reward-lab/) · [Algorithms](https://stablegradients.github.io/reward-lab/#algorithms/basics) · [original project](https://github.com/ayushnangia/reward-lab)

This fork trims the visualizer to a comparison of **RLOO, GRPO, TailRL, MaxRL and PKPO** on four starting shapes (bell, uniform, long right tail, binary 5% success) with the Adam optimizer, over a dataset of prompts with prompt-specific answer sets (a contextual bandit). `core.js` still implements and tests all ten methods; the sound, speed, shape-editor and video-export features of the original are removed.

Pick a starting distribution and the number of **answer sets**, then press **Play** or **Next step**. The dataset holds 262,144 prompts, each with 21 answers at fixed rewards; which answers score well differs between prompts. Prompts with the same answer set share one policy over reward levels, prompts with different sets share nothing, and every prompt starts from the same reward distribution. Each update draws 256 prompts and 32 answers per prompt, weights each prompt's group with the algorithm's rule, and averages the gradient for every policy in the batch (one epoch is 1,024 updates). With every prompt distinct, each is visited about once per epoch and the aggregate barely moves; fewer answer sets mean more transfer. At most 4,096 answer sets are simulated and stand in for the rest, which is exact in distribution because the sets are exchangeable. Charts show averages over prompts: mean reward, per-prompt entropy and per-prompt expected best-of-k. Playback runs at a fixed 8× (150 ms per update) for 1,000 updates by default. **Previous step** moves back through recorded updates; going forward replays them before computing another update. **Advanced** holds the algorithm ticks, reward transformations, judge errors, prompts per update, answers per prompt, learning rate, seed and run length; every change applies at once. Hover either step chart to move all three charts to any recorded update, as in a wandb panel.

The **System / Light / Dark** control follows the device theme by default and remembers a manual choice. Both pages, equations and native controls use the selected palette.

Reward positions stay fixed; bar height is probability, on a shared 0–100% scale. The gray bars retain the starting distribution. Replay uses recorded updates. A run stops at its chosen horizon.

**Learning curves** below the distributions plot every selected algorithm through the displayed update: mean original reward and policy entropy over updates, and the exact expected best-of-k reward for k from 1 to 64 at the displayed update. Hovering a step chart moves a shared cursor: the readouts and the best-of-k chart follow the hovered update. Replay moves the curves with the charts.

The Algorithms page starts with policies, rewards, baselines and gradients. Each method has a worked three-answer group: follow the scores, weights and changed probabilities with Previous step / Next step, or change the scores and reward map. These calculations use the existing numerical rules with a fixed balanced group; the main visualizer samples random groups. Chapters also explain score transformations, the relation to language models, paper sources and three runnable experiments. GRPO retains the clipping figure for both advantage signs.

## What the bars mean

This is a one-state, one-step categorical bandit with 21 actions. Each action has a fixed original reward in [0, 1] and its own adjustable logit. Softmax turns finite logits into positive probabilities; zero-support actions are explicitly masked with negative-infinity base logits. The bell-shaped preset is discrete and bounded. Initial support smoothing adds f to every starting probability and renormalizes, making an absent action's probability f / (1 + 21f). It is applied once, not after every update.

Sorting rearranges unchanged values. Policy learning changes probabilities. A reward map changes training scores; all plotted rewards and evaluation metrics stay on the original scale. Group normal scores do not make the policy distribution Gaussian.

A language model shares parameters across tokens and prompts. A reward histogram does not determine its gradients, parameter coupling, generalization, or training trajectory. There is no text generation, token-level aggregation, temporal credit assignment, entropy bonus or reference-policy KL penalty here. These are mechanism demonstrations, not full training replications or a performance ranking. Equal sample counts do not imply equal computation; the learning rates are not tuned separately for a benchmark.

With a clean judge and a fixed strictly increasing map, expected reward, expected best-of-k and finite TailRL all favor the highest-reward action with initial support. This unrestricted bandit does not force the tradeoffs caused by shared parameters, multiple prompts or regularization. MaxRL instead treats every action meeting its success threshold equally. This is a statement about objectives, not a convergence guarantee for sampled finite-step updates.

Displayed metrics are exact sums under the current policy, using original rewards even when the judge is corrupted. Expected best-of-k assumes independent draws and an oracle selector. The interface's k control sets both PKPO's training target and the k used by the cards' expected best-of-k metric; the best-of-k chart always spans k = 1 to 64. Hold k fixed across comparisons.

## Methods

| Method                                                            | Implemented mechanism                                                                                               |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| [GRPO](https://arxiv.org/abs/2402.03300)                          | Group-standardized advantages inside a clipped multi-epoch surrogate; reference KL coefficient β = 0                |
| [PPO](https://arxiv.org/abs/1707.06347)                           | Clipped probability-ratio surrogate, repeated epochs and learned scalar baseline                                    |
| [TailRL](https://arxiv.org/abs/2609.02987)                        | Centered finite-group tail weights; with n samples the mean-gradient convention implements order T = n − 1          |
| [RLOO](https://arxiv.org/abs/2402.14740)                          | Leave-one-out baseline and a categorical policy-gradient step                                                       |
| [MaxRL](https://arxiv.org/abs/2602.02710)                         | Binary judged success ≥ 0.9, centered and divided by the group success rate; no-success groups give zero weights    |
| [PKPO](https://arxiv.org/abs/2505.15201)                          | Continuous best-of-k leave-one-out-minus-one estimator, equation 33; k = 1 uses RLOO                                |
| [REINFORCE](https://link.springer.com/article/10.1007/BF00992696) | Sampled policy gradient without a baseline                                                                          |
| [A2C](https://arxiv.org/abs/1602.01783)                           | Synchronous one-step actor–critic; the linked paper supplies actor–critic foundations, not an A2C replication claim |
| [TRPO](https://arxiv.org/abs/1502.05477)                          | Categorical Fisher, conjugate gradients, exact KL evaluation and backtracking                                       |
| Elite                                                             | Top-20% likelihood heuristic with cutoff ties; not a full cross-entropy method                                      |

MaxRL bypasses continuous reward maps. PPO and GRPO reuse groups for multiple epochs. TRPO uses its KL budget instead of the shared actor learning rate. The other methods share one optimizer: Adam (β = 0.9 / 0.999, ε = 1e-8, bias-corrected, one state per method); `core.js` also supports a plain SGD step for tests. Adam divides each parameter's step by its running gradient scale, so a common rescaling of the rewards does not change the trajectory; with rewards exactly 0 and 1 (the binary preset) TailRL's and MaxRL's weight vectors are identical, so the two coincide under any optimizer; with two other levels on opposite sides of MaxRL's 0.9 threshold, such as 0.2 and 1, they differ by the reward gap under plain SGD but coincide under Adam. A group whose weights are all zero contributes no gradient; under Adam the decaying momentum from earlier groups can still move the policy for a few updates, under SGD nothing moves. All methods begin with the same distribution and use shared random uniforms. Once policies differ, they can sample different outcomes. Changing the visible methods does not change the underlying runs.

## Scientific review · 11 September 2026

The original interface, all ten algorithm chapters, their worked examples and experiment descriptions were checked against the numerical implementation and the linked sources. This includes dynamic status text, metric definitions, reward-map previews, audio labels, image captions and implementation caveats. Archived pages and saved sweeps are outside this review.

Source checks: REINFORCE's score-function estimator; RLOO §2.3; Mnih et al.'s terminal actor–critic specialization; PPO §3, equation 7; DeepSeekMath §§4.1.1–4.1.2; TRPO §§4–6 and Appendix C; MaxRL §4.3, Algorithm 1; TailRL §4 and Appendix C's centered estimator and mean-loss convention; PKPO §4.3, equation 33. The Elite link is context for quantile selection, not a claim that this heuristic reproduces CEM.

`verify.js` enumerates every four-sample group on a three-action policy and compares expected REINFORCE, A2C, RLOO, MaxRL, TailRL and PKPO gradients with numerical derivatives of their stated fixed-reward objectives. `verify-classics.js` checks clipping derivatives, active versus inactive clipping, repeated epochs, critic ordering and accepted TRPO steps. These finite checks do not establish LLM performance or convergence. GRPO's random group normalization is not labeled an unbiased expected-reward estimator.

Reward maps follow the formulas in `core.js`. [Box–Cox](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html) and [Yeo–Johnson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html) are additionally rescaled to [0,1] here, with a user-selected λ. Group normal scores use tied midranks; frozen scores use a clipped initial mid-CDF. None guarantees a Gaussian distribution for discrete rewards.

This is an implementation and source review, not independent peer review or a validation of the papers' empirical results.

## Run and develop

```sh
git clone https://github.com/ayushnangia/reward-lab.git
cd reward-lab
python3 server.py --open-browser
```

Open http://127.0.0.1:8766. Python's standard library serves the local files; no install or build is needed to use them. Everything runs in the browser. Stop the server with Control-C.

```sh
npm test
npm ci
npx playwright install chromium
npm run test:browser  # local server running on port 8766
node scripts/render-visualizer-preview.cjs
npm run build
```

Equations are authored in LaTeX and rendered with KaTeX, including accessible MathML. Its renderer, fonts and MIT license are bundled in `assets/katex/`, so equations also work locally without a CDN.

Numerical tests need only Node.js. Browser tests and preview generation use Playwright. `CHROME_PATH` can select an installed browser; `PLAYWRIGHT_MODULE` can select an existing Playwright installation. `QA_URL` overrides the browser test's base URL. Screenshots and test exports go into ignored `.qa/`.

Checks include exact finite-batch expectations, probability normalization and support, derivatives, clipping for both advantage signs, actual GRPO clipping behavior, TRPO's nonlinear KL constraint, zero-signal groups, transforms and ties. Browser checks cover all method pages from 320 to 1440 pixels, actual plotted state, replay, bounded playback, opt-in sound, validation, shared setups and PNG export. Lesson checks cover all ten worked calculations, softmax probabilities, affine maps, ties, navigation and every experiment button.

| Current file                       | Purpose                                                         |
| ---------------------------------- | --------------------------------------------------------------- |
| `index.html` / `rho.css`           | Two-page structure and responsive styling                       |
| `visualizer.js`                    | Playback, SVG charts, settings and replay                       |
| `curves.js` / `verify-curves.js`   | Learning-curve and best-of-k charts with a shared hover cursor  |
| `theme.js`                         | Appearance preference                                           |
| `algorithms.js`                    | Explanations, equations, implementation limits and edge cases   |
| `core.js`                          | Policies, reward maps, advantages, optimizers and exact metrics |
| `config.js` / `content.js`         | Shared validation, names and paper links                        |
| `verify*.js` / `qa-visualizer.cjs` | Numerical and browser checks                                    |
| `scripts/build.cjs`                | Explicit public asset list and cache versions                   |

To add a method, put its numerical rule in `core.js`, add names and a paper source in `content.js`, and an explanation in `algorithms.js`. Add a numerical check that distinguishes the rule from an incorrect implementation. The guide and comparison selector use this data directly.

Older research-note, output-viewer and sweep sources remain in the repository for reference; the current entry point and public build do not load them. Their existing numerical checks remain runnable. The old saved sweep predates the clipped GRPO update and is not evidence about the current visualizer.

## Writing style

Use literal headings and direct explanations. Avoid decorative taglines, metaphorical hooks and promotional filler. Use typography, spacing and layout for the visual style. Preserve the technical detail and worked examples.

## Hosting and design

`npm run build` replaces the generated `docs/` directory with only the current public assets. GitHub Pages serves `main:/docs`. This app needs static hosting, not a GPU or training server. A [Hugging Face static Space](https://huggingface.co/docs/hub/spaces-sdks-static) could serve the same generated files with `sdk: static` and `app_file: index.html` in its README configuration; it is optional.

The repeated comparison panels draw on [Mike Bostock's algorithm visualizations](https://bost.ocks.org/mike/algorithms/). The equation controls follow the explanatory approach of [Distill's momentum article](https://distill.pub/2017/momentum/). The [TailRL project page](https://zanette-labs.github.io/TailRL-website/) also informed the research presentation. The implementation and graphics are original; there is no affiliation.

MIT license. Contributions should preserve the distinction between a mathematical mechanism, a finite demonstration and an empirical result.
