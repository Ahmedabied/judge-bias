# judge-bias

Teams grade model answers with another model. If the judge quietly favours answers written by its own model, every
comparison that includes that model is tilted. judge-bias measures that tilt against a panel with no stake, with an
interval, and separates it from a judge that is merely noisy.

## The number

On the published MT-Bench human judgments (LMSYS, CC-BY-4.0), GPT-4 judged pairs of answers from six models, and
so did human experts. GPT-4 is one of the six. On the comparisons that include GPT-4's own answer:

| | GPT-4 as judge picks GPT-4 | human experts pick GPT-4 | self-preference | 95% interval |
|---|---|---|---|---|
| raw | 0.853 | 0.732 | **+0.120** | +0.085 to +0.155 |
| adjusted for how GPT-4 follows the humans elsewhere (k = 0.86) | | | **+0.152** | +0.116 to +0.188 |

290 comparisons over 80 questions; ties count half; all human judges pooled. It holds when cut other ways:

| slice | raw | adjusted |
|---|---|---|
| first turns only | +0.100 (+0.045 to +0.151) | +0.164 (+0.114 to +0.213) |
| second turns only | +0.138 (+0.090 to +0.186) | +0.205 (+0.159 to +0.253) |
| expert judges only, without the paper's authors | +0.118 (+0.084 to +0.154) | +0.163 (+0.129 to +0.198) |

So when GPT-4 grades a comparison that includes its own answer, it sides with itself about 12 to 15 points more
often than human experts do. The MT-Bench paper reported a similar raw gap; this adds the human yardstick, the
interval and the noise control. Reproduce it in a few seconds:

```sh
git clone https://github.com/Ahmedabied/judge-bias && cd judge-bias
python3 -m venv .venv && . .venv/bin/activate && pip install -e '.[public]'
judge-bias public      # downloads 1.4 MB, checks two pinned hashes, prints the table
```

CI runs the same command and fails if the number moves.

## Why the adjustment matters

`judge-bias demo` runs scripted models with no download: one judge always picks itself, one flips a coin, one is
fair, and two panel judges answer nothing. Alpha and bravo write the weaker answers:

| model | picks itself | panel picks it | raw | adjusted | 95% interval |
|---|---|---|---|---|---|
| alpha (prefers itself) | 1.000 | 0.287 | +0.713 | +0.713 | +0.637 to +0.787 |
| bravo (coin flip) | 0.531 | 0.319 | +0.212 | +0.034 | -0.032 to +0.098 |
| charlie (fair) | 0.894 | 0.894 | +0.000 | +0.000 | +0.000 to +0.000 |

The raw number accuses the coin flip. A judge that cannot tell answers apart lands near 50%, and when its own
model writes the weaker answers, 50% is more than the panel gives it. The adjusted number does not fall for that.
For GPT-4 the adjustment goes the other way: it follows the humans a little less sharply than they judge
(k = 0.86), so its lean toward itself is larger than the raw gap shows.

## Run it on your own models

You need [Ollama](https://ollama.com) and at least four models: three that answer and judge, and one or more that
only judge (the panel). The questions are the 80 first turns of MT-Bench (Apache-2.0, from the FastChat
repository), downloaded at run time and checked against a pinned sha256. Any models work; for example:

```sh
judge-bias run --answerers qwen2.5:1.5b,gemma2:2b,llama3.2:1b \
  --judges qwen2.5:1.5b,gemma2:2b,llama3.2:1b,granite3.3:2b,smollm2:1.7b --out results/run
```

Small models still need host memory: about 1.5 GB of RAM per loaded model even when it runs on the GPU. A full run
of five small models has not been published here yet; the local path is tested with scripted models.

Every answer and verdict is appended to JSON lines files as it arrives, so a stopped run picks up where it left
off. `judge-bias report results/run --answerers ...` re-analyses saved verdicts. Decoding is greedy with a fixed
seed. `--host` points at another local Ollama; remote hosts are refused.

## How it is measured

1. Each answer model answers every question.
2. Each judge compares every pair of answers twice, once in each order, and replies A or B. A judge that always
   says A ends up at exactly 50% for every model, so a first-slot habit cannot look like self-preference. It is
   reported on its own.
3. The panel is the judges that answer nothing: they have no stake in any pair.
4. **raw**: how often model X, as a judge, picks its own answer, minus how often the panel picks it.
5. **adjusted**: on the pairs where X has no stake (two other models), fit how closely X follows the panel,
   `p - 0.5 = k (r - 0.5)`. k is 1 for a judge that agrees with the panel and near 0 for a coin flip. Without
   self-preference X should pick itself at `0.5 + k (r - 0.5)`; adjusted is how far above that it lands, averaged.
   It needs three or more answer models and at least one panel judge.
6. Intervals: questions resampled with replacement 2,000 times, everything recomputed each time, 95% percentile.

Replies are parsed strictly. The letter has to lead the reply ("A", "B.", "Answer: B", "The better answer is A").
"A better answer is B" counts as unreadable, not as A, and unreadable verdicts are counted per judge, not guessed.

## Limits

- **Self-preference or style preference?** A judge may favour answers that sound like its own family's training,
  not its own answers as such. This measures the tilt either way; it cannot say why.
- **The adjustment is a model.** It assumes a judge relates to the panel the same way on its own pairs as on
  others' pairs. A panel that shares a bias with one model hides that bias.
- **One judge model, 2023 vintage.** The published number is GPT-4 (as of the MT-Bench paper) on 80 questions.
  Newer judges need their own run; the local path above is for that.
- **Humans are a panel, not the truth.** Experts disagree with each other too; they are the yardstick because they
  have no stake, not because they are always right.
- **The judge prompt is fixed** (`JUDGE_TEMPLATE` in [core.py](src/judge_bias/core.py)). Other prompts, rubrics or
  scores out of 10 can change the result.
- **Answers are capped** at 400 tokens by default, the same cap for every model.

## Development

```sh
pip install -e '.[test,public]'
pytest -q              # 32 tests, about 3 s
```

Tests use scripted models: a self-biased judge is found, a coin-flip judge is not blamed, a first-slot judge
cancels out, unreadable replies are counted, ties count half, a stopped run resumes without calling a model again,
and downloads are refused if a hash changes. CI also reproduces the published number.

Built with AI pair programming.

## Licence

MIT. Nothing from MT-Bench is bundled: the questions are downloaded from the FastChat repository (Apache-2.0) and
the judgments from lmsys/mt_bench_human_judgments on Hugging Face (CC-BY-4.0), both checked against pinned hashes.
