# Portfolio Validation Checklist

Before presenting the project as a 9–9.5/10 portfolio project:

- [ ] `python -m pytest -q` passes.
- [ ] `python pipeline.py <real-project.zip>` completes.
- [ ] `output/generation_manifest.json` is created.
- [ ] Generated TDD contains `EV-*` evidence IDs.
- [ ] Quality gate reports PASS for a clean sample.
- [ ] A deliberately unsupported claim is marked "Not determined..." or rejected.
- [ ] A malicious ZIP traversal test is rejected.
- [ ] Render `/health` returns HTTP 200.
- [ ] Render deployment uses `gunicorn app:app`.
- [ ] README explains evidence-grounding and limitations.
- [ ] At least one generated TDD has been manually spot-checked against source.
