import os
import subprocess
from pprint import pprint

if __name__ == "__main__":
    print("Running Module Tests\n ------------------------------")
    directory = os.path.dirname(__file__)
    filenames = [
        f
        for f in os.listdir(directory)
        if f.endswith(".py") and f not in ["__main__.py", "__init__.py"]
    ]
    resultset = []
    for filename in filenames:
        results = subprocess.run(
            [
                "python",
                "-m",
                "doctest",
                filename,
            ],
            capture_output=True,
            text=True,
        )
        if results.stdout != "":
            pprint(results.stdout)
            resultset.append(results)

    if resultset == []:
        print("All module level tests passed!")
