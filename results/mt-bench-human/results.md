# Results

- data: lmsys/mt_bench_human_judgments at f7d2896d2cc5 (CC-BY-4.0)
- verdicts: 5755; each turn compared with the same turn; ties count half; all human judges pooled as the panel

## Self-preference

| model | comparisons | picks itself | panel picks it | raw | 95% interval | follows the panel (k) | adjusted | 95% interval |
|---|---|---|---|---|---|---|---|---|
| gpt-4 | 578 | 0.852 | 0.733 | +0.119 | +0.083 to +0.154 | 1.104 | +0.095 | +0.049 to +0.139 |

## Judges

| judge | verdicts | unreadable | ties | picks the first slot | same winner in both orders |
|---|---|---|---|---|---|
| gpt-4 | 2400 | 0 | 600 | n/a | n/a |
| humans | 3355 | 0 | 780 | n/a | n/a |
