import os
import subprocess
from pprint import pprint

if __name__ == "__main__":
    print("Running Module Tests\n ------------------------------")
    mainpath = os.path.dirname(__file__)
    directory = os.path.join(mainpath, "pyamda")
    filenames = [ f for f in os.listdir(directory) if f.endswith(".py")]
    for filename in filenames:
        results = subprocess.run(["python", "-m", "doctest", f"pyamda/{filename}"], capture_output=True, text=True)
        if results.stdout != "": pprint(results.stdout)
