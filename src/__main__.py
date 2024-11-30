import os
import doctest
import importlib.util
import sys


def run_doctests_in_directory(directory):
    """
    Runs doctests for all .py files in the given directory, excluding __main__.py.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".py") and filename != "__main__.py":
            module_name = filename[:-3]  # Strip the .py extension
            file_path = os.path.join(directory, filename)

            try:
                # Dynamically import the module
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                assert spec is not None
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                loader = spec.loader
                assert loader is not None
                loader.exec_module(module)

                # Run doctests on the module
                print(f"Running doctests in {filename}...")
                doctest.testmod(module, verbose=True)
            except Exception as e:
                print(f"Failed to run doctests in {filename}: {e}")


if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    run_doctests_in_directory(current_directory)
