# Tiny RAG: plug an app into evalX in one minute

The smallest complete example of the evalX loop: a toy RAG app, four golden
cases, a baseline, one change that improves the app and one that breaks it.
It runs offline. No API key, no model download.

## The plug

evalX calls exactly one function in your app:

```python
def answer(question: str) -> str:
    return generate(question, retrieve(question))
```

You point evalX at it with `--target module:function`, here
`examples.tiny_rag.app:answer`. Retrieval, prompts and the model stay
inside your app; evalX only sees the question going in and the answer
coming out. Any app that can be wrapped in that one function can be
graded, whether it's plain Python, an HTTP service
([`examples/aerointel_app.py`](https://github.com/SujahathMSM/evalX/blob/main/examples/aerointel_app.py))
or an OpenAI-compatible endpoint
([`examples/openai_compatible_app.py`](https://github.com/SujahathMSM/evalX/blob/main/examples/openai_compatible_app.py)).

The "model" is switched with the `TINY_RAG_MODEL` environment variable:

| Model | What it does |
| --- | --- |
| `v1` (default) | quotes the whole retrieved passage |
| `v2` | keeps only the first sentence |
| `v3` | like v2, but answers off-topic questions instead of refusing |

## Walkthrough

Run these from the root of a clone, after `pip install -e .`.

**1. Grade v1 and pin it as the baseline.**

```bash
evalx run --suite examples/tiny_rag/cases.yaml --target examples.tiny_rag.app:answer
evalx baseline pin latest --name tiny-rag
```

```
torque-value     0.50
tire-gas         1.00
cable-rule       1.00
refuse-offtopic  1.00
Mean score: 0.875 over 4 cases
RESULT: PASS
```

`torque-value` only half passes: v1 gives the right value (25 in-lb) but
also repeats the manual's warning "Never use **ft-lb** values", and the
`not_contains: ft-lb` check catches the wrong unit in the answer.

**2. Switch to v2 and compare against the baseline.**

```bash
TINY_RAG_MODEL=v2 evalx run --suite examples/tiny_rag/cases.yaml \
  --target examples.tiny_rag.app:answer --baseline tiny-rag
evalx compare latest-1 latest
```

```
No case regressed past the threshold.
RESULT: PASS

torque-value  0.50 -> 1.00  +0.50
Mean: 0.875 -> 1.000 (+0.125) | 1 improved, 0 regressed, 3 unchanged
```

**3. Switch to v3.**

```bash
TINY_RAG_MODEL=v3 evalx run --suite examples/tiny_rag/cases.yaml \
  --target examples.tiny_rag.app:answer --baseline tiny-rag
```

```
Infra gate passed: 0/4 cases errored (allowed rate 0.00)
Mean gate passed: mean 0.750 >= 0.0
  REGRESSED refuse-offtopic: 1.00 -> 0.00 (drop 1.00)
REGRESSION GATE FAILED: 1 case(s) dropped past 0.2

RESULT: FAIL (regression gate)
```

v3 answers "Paris is the capital of France." instead of refusing. It even
fixes `torque-value`, so a quick look would call it an upgrade. Each gate
reports on its own line: no errors (infra), a mean of 0.750 above the
default bar of 0.0 (mean), but one case fell from 1.00 to 0.00
(regression). The last line is the verdict for the whole run, and the
command exits 1. In CI that fails the build. In PowerShell, `$LASTEXITCODE`
shows the 1.

On Windows PowerShell, set the model with `$env:TINY_RAG_MODEL = "v2"` on
its own line before the `evalx run` command, and
`Remove-Item Env:TINY_RAG_MODEL` to go back to v1.

## Make it yours

1. Replace `answer()` in `app.py` with a call into your own app.
2. Replace the questions in `cases.yaml` with ones your users ask, each with
   the checks a correct answer must pass. The README's "Check types"
   section lists them.
3. Run, pin a baseline, and add `--baseline` to the run in CI.
