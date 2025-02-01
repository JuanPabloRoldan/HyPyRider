# Tests Directory

This directory contains unit tests for the **HyPyRider** project, ensuring the correctness of functions and classes across the codebase. The tests are written using [`pytest`](https://docs.pytest.org/).

## Structure
Currently, this directory contains two test scripts, each covering multiple functions with `assert` statements to verify expected outputs. The goal is to expand the test suite over time by adding more test cases and coverage for additional functionalities.

## Running Tests
To execute the tests, navigate to the project's root directory and use the following command:

```sh
PYTHONPATH=$(pwd) pytest
```

### Explanation:
- `PYTHONPATH=$(pwd)`: Ensures that Python recognizes the project’s source directory (`src/` or equivalent) when running tests.
- `pytest`: Runs all test scripts in the `tests/` directory automatically.

## Importing Modules Correctly
To maintain consistency and follow industry standards, always use absolute imports. Since the project's source code is inside the `src/` directory, ensure that test scripts import modules using the following format:

```python
from src.some_module import some_function
```

Avoid using relative imports or modifying `PYTHONPATH` beyond setting it to the project root.

## Adding More Tests
To contribute additional tests:
1. Create a new test script (e.g., `test_new_feature.py`) inside the `tests/` directory.
2. Use the following structure for test functions:

```python
import pytest
from src.some_module import some_class

@pytest.fixture
def initializer():
    return ClassInstance()

def test_some_function():
    output = initializer.some_function(input)
    assert output == expected_output
```

3. Run `pytest` again to validate the new test cases.

## Dependencies
Ensure `pytest` is installed before running tests:

```sh
pip install pytest
```

## Best Practices
- Keep test files in the `tests/` directory, separate from the `src/` directory.
- Follow the `test_*.py` naming convention.
- Use absolute imports (`from src.module import function`) for clarity.
- Each test function should start with `test_` to be automatically discovered by `pytest`.
- Use fixtures and parametrization for better test coverage.

---
Expanding and maintaining the test suite will help catch regressions early and improve code reliability. 🚀

