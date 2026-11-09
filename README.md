# judge-bias

Small models are often used to grade other models' answers. judge-bias measures whether such a judge favours
answers written by its own model, using free local models and a correction for judges that are just noisy.

Work in progress: results from local models are being added.

```sh
python3 -m venv .venv && . .venv/bin/activate && pip install -e .
judge-bias demo
```

Built with AI pair programming.

## Licence

MIT
