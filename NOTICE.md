# Notices

## `t2agent/text.py`

`t2agent/text.py` is not used by the forecaster: no module imports it, and the Final configuration declares no model.
It is a small client for the competition's hosted language model, adapted from `baselines/reasoning_agent.py` in
[Agenthon-2026/track2-forecasting-public](https://github.com/Agenthon-2026/track2-forecasting-public) (MIT licence,
reproduced below). It reads its endpoint and credentials only from environment variables and contains no keys. The file is
kept so that `t2agent/` stays byte-identical to the Final entry.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Baseline (M0)

The text-blind baseline that the forecaster starts from was re-implemented from the organizers' published description
(`docs/M0-BASELINE.md` in the same repository). No organizer code other than the file above is included.
