# Tests for Strider

Comprehensive test suite for the clinical gait analysis pipeline.

## Structure

```
tests/
├── unit/              # Unit tests (fast, no video I/O)
│   ├── test_quartile_engine.py
│   ├── test_homography.py        (TODO: Phase 1)
│   ├── test_foot_strike.py       (TODO: Phase 1)
│   ├── test_phase_detector.py    (TODO: Phase 1)
│   └── conftest.py               # Shared fixtures
├── integration/       # Integration tests (slower, with synthetic data)
│   └── test_pipeline_synthetic.py (TODO: Phase 1)
├── fixtures/         # Test data generators
│   ├── conftest.py    (TODO: Phase 1)
│   └── synthetic_gait.py (TODO: Phase 1)
└── data/             # Test video files
    ├── patient_*.mp4  (your test videos)
    └── healthy_*.mp4
```

## Running Tests

### All tests
```bash
pytest tests/
```

### Unit tests only (fast)
```bash
pytest tests/unit/ -v
```

### Specific test file
```bash
pytest tests/unit/test_quartile_engine.py -v
```

### With coverage
```bash
pytest tests/ --cov=strider --cov-report=html
```

### Integration tests (slower)
```bash
pytest tests/integration/ -v
```

## Test Data

Place your test videos in `tests/data/`:
- Patient videos: `patient_001.mp4`, `patient_002.mp4`, etc.
- Healthy control videos: `healthy_001.mp4`, `healthy_002.mp4`, etc.

Tests will skip integration tests if video files are missing.

## Markers

Tests are marked for easy filtering:

```bash
# Run only unit tests (fast)
pytest -m unit tests/

# Run only integration tests
pytest -m integration tests/

# Skip slow tests
pytest tests/ -m "not slow"
```

## Contributing Tests

When implementing a new module:

1. Write unit tests FIRST (TDD approach)
2. Test with synthetic data (no video I/O)
3. Then implement the module
4. Verify tests pass

Example:
```python
# test_my_module.py
def test_my_function():
    from stride.mymodule import my_function
    result = my_function(input_data)
    assert result == expected_output
```

## Validation Strategy (Phase 6)

Once all Phase 1-5 modules are complete:

1. Validate against synthetic ground truth
2. Compare to manual annotations on real videos
3. Calculate ICC for reliability
4. Document accuracy bounds

See [ROADMAP.md](../ROADMAP.md#phase-6-research-grade-robustness) for details.
