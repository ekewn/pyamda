import os
import doctest
import importlib.util as U
import sys
from typing import Optional


def run_doctests(directory: str, filename: str) -> Optional[doctest.TestResults]:
    module_name = filename[:-3]  # Strip the .py extension
    file_path = os.path.join(directory, filename)

    spec = U.spec_from_file_location(module_name, file_path)
    assert spec is not None, f"{module_name}, {file_path}"

    module = U.module_from_spec(spec)
    sys.modules[module_name] = module

    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)

    return doctest.testmod(module)


if __name__ == "__main__":
    print("Running Module Tests\n ------------------------------")
    directory = os.path.dirname(os.path.abspath(__file__))
    filenames = [
        f
        for f in os.listdir(directory)
        if f.endswith(".py") and f not in {"__main__.py", "__init__.py"}
    ]
    for filename in filenames:
        results = run_doctests(directory, filename)
        (
            print(f"All tests for {filename} passed!")
            if results is not None and results.failed == 0
            else print()
        )
