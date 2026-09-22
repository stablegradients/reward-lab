/* LaTeX, lessons and finite experiments. Numerical rules live in core.js. */
const RhoAlgorithms = {
  rloo: {
    name: "REINFORCE Leave-One-Out",
    short: "Other-sample baseline",
    lead: "Judge an answer relative to the other attempts at the same problem. Those other scores provide a baseline without training a critic.",
    equation: "A_i = r_i - \\frac{\\sum_{j\\ne i}r_j}{n-1}",
    terms: [
      ["n", "Number of sampled outcomes in this group."],
      ["A_i", "Reward relative to the other samples."],
      ["r_i", "The training score of answer i."],
      [
        "j",
        "An index over the other samples; \\(j\\ne i\\) excludes this answer.",
      ],
    ],
    here: "One categorical policy-gradient step per group. A fixed nonlinear reward map changes the target from expected reward to expected mapped reward.",
    why: [
      "A reward of 0.5 might be encouraging on a difficult problem and disappointing on an easy one. A baseline supplies that context. An advantage is the reward minus this comparison value.",
      "RLOO uses the other answers in the group as the comparison. Leaving the current answer out matters: otherwise that answer would help determine the baseline being subtracted from itself.",
    ],
    read: [
      "For answer \\(i\\), sum the other \\(n-1\\) rewards and divide by \\(n-1\\). Subtract that average from \\(r_i\\). A negative weight is allowed; it is not a negative probability.",
      "For independent samples from the same policy and a fixed reward function, the other-sample baseline preserves the expected policy gradient. The resulting update still uses \\(A_i\\nabla_\\theta\\log\\pi_\\theta(y_i)\\).",
    ],
    shaping: [
      "A shared additive offset cancels exactly. Multiplying scores by a positive constant multiplies the advantages by that constant. A nonlinear map changes the relative gaps and therefore the expected-reward objective.",
      "Normal scores fitted separately to each group are a different case: an answer’s training score now depends on its companions. The usual fixed-reward unbiasedness argument no longer applies directly. No Gaussian reward assumption is needed.",
    ],
    transfer:
      "In sequence-level RL, the group contains multiple completions of one prompt. An easy prompt’s answers should not serve as the baseline for an unrelated hard prompt. RLOO removes the need for a learned baseline, but the extra completions still cost generation time.",
    citation: "Ahmadian et al. · §2.3, leave-one-out estimator",
    cases: [
      {
        title: "Too few rare successes",
        text: "Use a small group and a rare excellent outcome. Many groups contain only poor answers and produce no relative signal. Increase the group size and compare several seeds.",
        settings: {
          preset: "rare",
          n: 8,
        },
      },
      {
        title: "All scores tied",
        text: "No answer beats its companions, so every advantage is zero. Normalizing the tied scores cannot invent an ordering.",
        settings: {
          preset: "zero",
        },
      },
      {
        title: "A nonlinear score map",
        text: "Compare Identity and Square on two modes. Inspect the weights in a sampled group before interpreting the changed bars.",
        settings: {
          preset: "bimodal",
          transform: "square",
        },
      },
    ],
    comparison: ["rloo", "grpo", "tailrl"],
  },
  grpo: {
    name: "Group Relative Policy Optimization",
    short: "Group advantages + clipping",
    lead: "Compare several answers to the same prompt, then use their relative scores in a clipped update. GRPO gets its baseline from the group instead of a critic.",
    equation: "A_i = \\frac{r_i-\\mu}{\\sigma}",
    terms: [
      ["r_i", "The training reward for sample i."],
      [
        "\\mu,\\sigma",
        "Group mean and standard deviation; the variance is the mean squared deviation (divisor n).",
      ],
      [
        "A_i",
        "The sample’s advantage, held fixed during the optimization epochs.",
      ],
    ],
    here: "We apply the group advantages through a PPO-style clipped categorical surrogate. We add \\(10^{-8}\\) to \\(\\sigma\\) for numerical stability; equal-reward groups use zero advantages. The paper’s token averaging and reference-policy KL penalty are omitted (\\(\\beta=0\\)).",
    clipped: true,
    clippedEquation:
      "\\begin{aligned} L &= \\frac{1}{n}\\sum_{i=1}^{n}\\ell_i, \\\\[4pt] \\ell_i &= \\min\\!\\left(q_i A_i,\\widetilde{q}_i A_i\\right). \\end{aligned}",
    why: [
      "Subtracting the average distinguishes better from worse attempts. Dividing by the standard deviation expresses the difference in units of the group’s spread. This is an advantage calculation, not a recipe for changing scores into a Gaussian distribution.",
    ],
    read: [
      "Compute the group mean \\(\\mu\\), subtract it from each score, and divide by the group standard deviation \\(\\sigma\\). A weight of +1 means one group standard deviation above the mean.",
      "Keep these advantages fixed while optimizing the clipped surrogate. The original GRPO objective averages over tokens and includes a reference-policy KL penalty. Here each episode has one action, and the KL penalty is omitted.",
    ],
    shaping: [
      "For \\(f(r)=ar+b\\) with \\(a>0\\), the shift and scale cancel in the ideal standardized advantage. Our numerical stabilizer makes this approximate for extremely small spreads. Try the affine map in the worked example.",
      "Squaring scores preserves their order but changes relative gaps, so the standardized weights can change. A preceding z-score is largely redundant. The group-dependent divisor means this is not generally an unbiased estimate of the fixed expected-reward gradient.",
    ],
    transfer:
      "Real groups contain answers to the same prompt. Standardization can give a small reward difference on one prompt a comparable weight to a large difference on another. Whether that helps depends on what the score differences mean. Shared token parameters, reward reliability and the prompt mixture determine the resulting generalization.",
    citation:
      "DeepSeekMath · §§4.1.1–4.1.2, clipped objective and outcome advantages",
    cases: [
      {
        title: "Every answer ties",
        text: "The standard deviation is zero. This implementation explicitly uses zero advantages; adding numerical epsilon is not new information.",
        settings: {
          preset: "zero",
        },
      },
      {
        title: "Almost tied answers",
        text: "Compare GRPO with RLOO on a narrow score range. Inspect advantage magnitudes; a visible update need not imply a large quality difference.",
        settings: {
          preset: "narrow",
        },
      },
      {
        title: "A faulty judge",
        text: "A false high score can make a poor answer rank first in its group. Relative normalization cannot detect that judge error.",
        settings: {
          preset: "right",
          judge: "falsepositive",
        },
      },
    ],
    comparison: ["grpo", "rloo", "tailrl"],
  },
  tailrl: {
    name: "Tail-Likelihood Reinforcement Learning",
    short: "Tail-weighted gradient",
    lead: "Ask how likely an answer is to clear each reward threshold. Improving a threshold that few answers clear receives more weight.",
    equation:
      "J(\\theta)=\\int_0^1\\log\\operatorname{Pr}_{\\theta}(R>t)\\,\\mathrm{d}t",
    terms: [
      ["t", "A reward threshold."],
      [
        "\\operatorname{Pr}_{\\theta}(R>t)",
        "Probability of exceeding that threshold.",
      ],
      [
        "J",
        "Population objective. The visualizer uses a finite-group gradient estimator.",
      ],
      ["R", "The reward of a random answer."],
      ["\\theta", "The parameters determining answer probabilities."],
    ],
    here: "The displayed population integral uses rewards in \\([0,1]\\). With a fixed reward map, the centered estimator implements finite order \\(T=n-1\\) using \\(n\\) independent samples and a mean-gradient update. A group-fitted map changes that fixed-reward argument. Sorting computes weights; it does not reorder the policy’s bars.",
    why: [
      "Mean reward averages scores. TailRL averages log probabilities of exceeding reward thresholds. What a threshold means depends on how the reward is defined.",
      "If a threshold is cleared with probability \\(p\\), a small absolute improvement has local weight \\(1/p\\) inside \\(\\log p\\). Integrating over thresholds extends the log-success idea to graded rewards.",
    ],
    read: [
      "\\(\\Pr_\\theta(R>t)\\) is the probability mass to the right of threshold \\(t\\). The integral combines these log tail probabilities over rewards normalized to \\([0,1]\\).",
      "The browser uses a finite-group estimator. Sort the scores, divide each new score gap by how many samples survive above the previous level, and accumulate those increments. Center the accumulated values and multiply by group size before the mean-gradient update.",
    ],
    shaping: [
      "For a fixed smooth increasing map \\(f\\), rescaling the reward axis weights thresholds by \\(f'(t)\\). Changing relative slopes therefore changes the objective. The piecewise option is rescaled to keep endpoints at 0 and 1; its slope and jump controls are relative.",
      "A continuous increasing map of a fully supported interval cannot insert a literal support gap. It can change density and make a region look sparse. A jump in the score map can create a gap in score space; learning changes the probabilities separately.",
    ],
    transfer:
      "Thresholds come from the judge, not from the model’s internal reasoning. A rare false high score can receive strong weight. A hollow middle in a histogram does not show that intermediate reasoning skills disappeared: that requires tests on actual outputs and prompts.",
    citation: "TailRL · tail objective, Proposition 9 and Appendix F",
    cases: [
      {
        title: "A judge error in the tail",
        text: "Compare true poor-output probability with the training score. A rare high label may reward the wrong outcome.",
        settings: {
          preset: "right",
          judge: "falsepositive",
        },
      },
      {
        title: "No upper-tail support",
        text: "The population log objective is singular if an interval of thresholds has zero survival probability. Finite-group weights remain computable, but they cannot create missing support here.",
        settings: {
          preset: "missing",
        },
      },
      {
        title: "Different slopes",
        text: "Use the continuous piecewise map to shrink low-score gaps relative to high-score gaps. Inspect original-reward probabilities; stretching score gaps does not itself move probability mass.",
        settings: {
          preset: "spike",
          transform: "hinge",
          jump: 0,
        },
      },
    ],
    comparison: ["tailrl", "grpo", "maxrl"],
  },
  maxrl: {
    name: "Maximum Likelihood Reinforcement Learning",
    short: "Binary success weights",
    lead: "Treat an answer as a success or a failure. Scale its relative success label by how often the group succeeds, so a rare success receives a strong weight.",
    equation:
      "A_i = \\begin{cases} \\dfrac{b_i-\\widehat{p}}{\\widehat{p}}, & \\widehat{p}>0, \\\\[5pt] 0, & \\widehat{p}=0. \\end{cases}",
    terms: [
      ["b_i", "1 if the judged reward is at least 0.9; otherwise 0."],
      ["\\widehat{p}", "Fraction of successful samples in the group."],
      ["A_i", "The weight applied to this answer’s log-probability gradient."],
    ],
    here: "This finite-group binary rule bypasses the reward map. We use zero weights when the group has no successes; an all-success group also has zero centered weights. It is not the paper’s full LLM training system. With n samples, this centered-and-skipped estimator targets finite order T = n − 1; it is not an exact estimator of the infinite-order log objective.",
    why: [
      "Let \\(p\\) be the chance of a correct answer on one prompt. Maximizing \\(\\log p\\) puts more gradient weight on an equal absolute improvement when \\(p\\) is small, since its derivative is \\(1/p\\).",
      "For one unconstrained prompt, success probability and log success probability have the same maximizer. Across prompts sharing model parameters, weighting improvements differently can matter. MaxRL uses a finite-sample family related to this log-success objective.",
    ],
    read: [
      "Convert judged scores into binary labels \\(b_i\\). Here a score of at least 0.9 counts as success. Compute their mean \\(\\widehat p\\), then divide centered labels by that mean.",
      "For labels \\([0,0,1]\\), \\(\\widehat p=1/3\\), giving weights \\([-1,-1,2]\\). A no-success group must be handled separately because division by zero is undefined; this app skips it.",
    ],
    shaping: [
      "The success definition must stay meaningful. This app thresholds the judge’s original score and bypasses continuous reward maps. Selecting Square or an affine map in the worked example therefore leaves MaxRL unchanged.",
      "Turning a continuous task into pass/fail discards progress below the threshold. Going from 0.1 to 0.8 is useful according to the original scores, but both are failures under the chosen threshold. MaxRL does not fit a normal distribution.",
    ],
    transfer:
      "A correctness checker can supply binary labels for model completions. Difficulty-dependent weights can redistribute learning across prompts, but an unseen success still supplies no sampled success signal. A false positive is particularly consequential when genuine successes are rare.",
    citation:
      "Tajwar et al. · §4.3, Algorithm 1 (centered; no-success groups give zero weights)",
    cases: [
      {
        title: "No successful sample",
        text: "Try missing successes. Every judged score is below 0.9, so this rule has zero weights even when other methods distinguish partial progress.",
        settings: {
          preset: "missing",
        },
      },
      {
        title: "Rare successes",
        text: "Some groups contain no success; occasional groups do. Compare seeds and group sizes before calling a single jump an algorithm advantage.",
        settings: {
          preset: "rare",
          n: 8,
        },
      },
      {
        title: "False positives",
        text: "A mislabeled poor output can receive the rare-success weight. Inspect true poor-output probability, not just the binary training score.",
        settings: {
          preset: "right",
          judge: "falsepositive",
        },
      },
    ],
    comparison: ["maxrl", "tailrl", "rloo"],
  },
  pkpo: {
    name: "Pass@K Policy Optimization",
    short: "Best-of-k gradient",
    lead: "Train for the best result among several attempts. An answer is useful when it improves on what the other attempts would already provide.",
    equation:
      "\\begin{aligned} m_S &= \\max_{j\\in S}r_j, \\\\[4pt] A_i &= k\\,\\mathbb{E}_S\\!\\left[(r_i-m_S)_+\\right]. \\end{aligned}",
    terms: [
      ["k", "Target number of attempts, at most the group size here."],
      [
        "S",
        "A uniformly chosen subset of \\(k-1\\) other sample indices, excluding \\(i\\).",
      ],
      ["(x)_+", "\\(\\max(0,x)\\)."],
      ["\\mathbb{E}_S", "Average over those subsets."],
      ["r_i", "The training score of answer i."],
      ["m_S", "The best score in the chosen subset."],
      ["n", "The number of samples in the group."],
      ["A_i", "The weight used in the group-mean gradient."],
    ],
    here: "The continuous-reward leave-one-out-minus-one estimator from equation 33. For \\(k=1\\) we use RLOO. With fixed binary rewards the objective is pass@k; with fixed graded rewards it is expected maximum reward.",
    equationNote:
      "For \\(2\\le k\\le n\\); when \\(k=1\\), the visualizer uses RLOO.",
    why: [
      "If you can try \\(k\\) times and reliably select the best answer, the relevant target can be expected best-of-\\(k\\) reward. A merely adequate answer may add little when another attempt is already better.",
      "For binary correctness this is pass@\\(k\\), the chance that at least one attempt succeeds. For continuous scores the target here is the expected maximum score; selection quality remains an assumption.",
    ],
    read: [
      "Choose \\(k-1\\) other sample indices, called \\(S\\). Their best reward is \\(m_S\\). Measure how much answer \\(i\\) beats that best reward, using zero if it does not.",
      "Average over all eligible subsets and multiply by \\(k\\). Our gradient averages these weights over the group; the paper’s equation 33 folds that group-size factor into its coefficients. This is the continuous leave-one-out-minus-one estimator.",
    ],
    shaping: [
      "An increasing map preserves which member of a set is best, but changes the size of the improvement. Consequently it can change the best-of-k gradient. A shared offset cancels out of these differences for k greater than one.",
      "For k = 1 this app uses RLOO. Training and evaluation budgets are distinct concepts; this interface uses one k control for both. Keep the evaluation budget fixed when comparing runs. A group-fitted reward map no longer estimates best-of-k for a fixed scoring function.",
    ],
    transfer:
      "Retries help only when the system can afford them and can recognize a good answer. A learned judge may select the wrong one. Training for retries and training for one dependable response are different objectives; use both single-attempt and retry metrics when assessing transfer.",
    citation: "PKPO · §4.3, equation 33 (group-mean convention here)",
    cases: [
      {
        title: "Two modes, eight attempts",
        text: "Compare mean reward and expected best-of-8. A higher retry score does not by itself demonstrate better typical answers.",
        settings: {
          preset: "bimodal",
          k: 8,
        },
      },
      {
        title: "A rare excellent answer",
        text: "If the group misses the rare answer, it cannot credit that answer’s marginal contribution. Compare more samples and different seeds.",
        settings: {
          preset: "rare",
          n: 8,
          k: 4,
        },
      },
      {
        title: "A mistaken selector",
        text: "A false high score makes a poor answer appear to improve the set. Compare the judged objective with true-score metrics.",
        settings: {
          preset: "right",
          judge: "falsepositive",
          k: 4,
        },
      },
    ],
    comparison: ["pkpo", "rloo", "tailrl"],
  },
};
