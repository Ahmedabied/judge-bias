# Results

- data: lmsys/mt_bench_human_judgments at f7d2896d2cc5 (CC-BY-4.0)
- verdicts: 5755; both turns; ties count half; all human judges pooled as the panel

## Self-preference

| model | comparisons | picks itself | panel picks it | raw | 95% interval | follows the panel (k) | adjusted | 95% interval |
|---|---|---|---|---|---|---|---|---|
| gpt-4 | 290 | 0.853 | 0.732 | +0.120 | +0.085 to +0.155 | 0.863 | +0.152 | +0.116 to +0.188 |

## Judges

| judge | verdicts | unreadable | ties | picks the first slot | same winner in both orders |
|---|---|---|---|---|---|
| gpt-4 | 2400 | 0 | 600 | n/a | n/a |
| humans | 3355 | 0 | 780 | n/a | n/a |
