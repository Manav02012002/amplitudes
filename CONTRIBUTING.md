# Contributing

Thanks for contributing!

## Dev setup

```bash
pip install -e ".[dev]"
pre-commit install
pytest
```

## Style

- Format/lint: `ruff format` and `ruff check`
- Keep tests fast; mark long tests with `@pytest.mark.slow`.
