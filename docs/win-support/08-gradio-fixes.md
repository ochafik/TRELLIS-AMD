# Gradio Client Fixes

## Problem: TypeError in json_schema_to_python_type

When starting the Gradio app, an error occurs:

```
TypeError: argument of type 'bool' is not iterable
  File "gradio_client/utils.py", line 863, in get_type
    if "const" in schema:
```

### Root Cause

The `_json_schema_to_python_type` function doesn't handle boolean schemas. In JSON Schema, `additionalProperties` can be either:
- A schema object (dict) describing allowed additional properties
- `true` (boolean) meaning any additional properties allowed
- `false` (boolean) meaning no additional properties allowed

When `additionalProperties: true` is passed, the code tries to use `in` operator on a boolean.

### Solution: Patch gradio_client/utils.py

Add boolean handling at the start of `_json_schema_to_python_type`:

```python
def _json_schema_to_python_type(schema: Any, defs) -> str:
    """Convert the json schema into a python type hint"""
    if schema == {}:
        return "Any"
    # Handle boolean schemas (additionalProperties: true/false)
    if schema is True:
        return "Any"
    if schema is False:
        return "None"
    type_ = get_type(schema)
    # ... rest of function
```

### File Location

```
.venv/Lib/site-packages/gradio_client/utils.py
```

Around line 897.

### Note

This is a temporary fix in the installed package. The proper fix should be submitted upstream to the gradio-client project. The issue occurs with gradio-client 1.3.0 but may be fixed in newer versions.
