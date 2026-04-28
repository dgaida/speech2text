# Docstring Style Guide

We follow the **Google Style** for all docstrings in the project. This ensures consistency and clean generation of the API documentation.

## Example

```python
def function(arg1, arg2):
    """Short summary.

    Longer description if necessary.

    Args:
        arg1 (int): Description of the first argument.
        arg2 (str): Description of the second argument.

    Returns:
        bool: Description of the return value.

    Raises:
        ValueError: If arg1 is invalid.
    """
```

## Requirements

- Every public class, method, and function must have a docstring.  
- Types should be specified in the docstring even if type hints are used.  
- Coverage is enforced by `interrogate` (threshold 95%).  
