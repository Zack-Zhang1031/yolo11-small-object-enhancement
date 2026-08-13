# Repository Agent Instructions

1. Do not start a training job unless the user explicitly requests it; `--dry-run` is allowed for validation.
2. After changing a network graph, run `python scripts/build_all.py`.
3. After changing a module, run `pytest -q`.
4. Report metrics only with traceable experiment artifacts.
5. Do not modify global or environment `site-packages`.
6. Do not commit model weights, datasets, runs or large caches.
7. Keep YAML declarations consistent with instantiated modules.
8. Prefer the smallest focused change.
9. Run `git diff` after each completed task.
10. Confirm all relevant tests pass before pushing.
