# judge-bias

Teams grade model answers with another model, often a small local one. If the judge quietly favours answers written
by its own model, every comparison that includes that model is tilted. judge-bias measures that tilt on free local
models, with an interval, and separates it from a judge that is merely noisy.

## Try it in under a minute

```sh
git clone https://github.com/Ahmedabied/judge-bias && cd judge-bias
python3 -m venv .venv && . .venv/bin/activate && pip install -e .
judge-bias demo        # scripted models, no download, about a second
```

The demo has one judge that always picks itself, one that flips a coin, one that is fair, and two panel judges that
answer nothing. Its weaker models are alpha and bravo:

| model | picks itself | panel picks it | raw | adjusted | 95% interval |
|---|---|---|---|---|---|
| alpha (prefers itself) | 1.000 | 0.287 | +0.713 | +0.713 | +0.637 to +0.787 |
| bravo (coin flip) | 0.531 | 0.319 | +0.212 | +0.034 | -0.032 to +0.098 |
| charlie (fair) | 0.894 | 0.894 | +0.000 | +0.000 | +0.000 to +0.000 |

The raw number accuses the coin flip. A judge that cannot tell answers apart lands near 50%, and when its own
model writes the weaker answers, 50% is more than the panel gives it. The adjusted number does not fall for that.

## Run it on real models

You need [Ollama](https://ollama.com) and a few small models. The questions are the 80 first turns of MT-Bench
(Apache-2.0, from the FastChat repository), downloaded at run time and checked against a pinned sha256.

```sh
ollama pull qwen2.5:1.5b && ollama pull gemma2:2b && ollama pull llama3.2:3b     # answer and judge
ollama pull granite3.3:2b && ollama pull smollm2:1.7b                             # panel: judge only
judge-bias run --answerers qwen2.5:1.5b,gemma2:2b,llama3.2:3b \
  --judges qwen2.5:1.5b,gemma2:2b,llama3.2:3b,granite3.3:2b,smollm2:1.7b --out results/run
```

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
- **The panel is not ground truth.** It is the yardstick, and the adjustment assumes a judge relates to the panel
  the same way on its own pairs as on others' pairs. A panel that shares a bias with one model hides that bias.
- **Small models, small sample.** 80 questions, models up to 3B. Bigger judges may behave differently.
- **The judge prompt is fixed** (`JUDGE_TEMPLATE` in [core.py](src/judge_bias/core.py)). Other prompts, rubrics or
  scores out of 10 can change the result.
- **Answers are capped** at 400 tokens by default, the same cap for every model.

## Development

```sh
pip install -e '.[test]'
pytest -q
```

Tests use scripted models: a self-biased judge is found, a coin-flip judge is not blamed, a first-slot judge
cancels out, unreadable replies are counted, a stopped run resumes without calling a model again, and the prompt
file is refused if its hash changes.

Built with AI pair programming.

## Licence

MIT. The MT-Bench questions are not bundled; they are downloaded from the FastChat repository (Apache-2.0).
