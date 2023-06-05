from datetime import datetime
from pathlib import Path

from tango.version import VERSION


def main():
    citation = Path("CITATION.cff")

    with citation.open() as f:
        lines = f.readlines()

    for i in range(len(lines)):
        line = lines[i]
        if line.startswith("version:"):
            lines[i] = f'version: "{VERSION}"\n'
        elif line.startswith("date-released:"):
            lines[i] = f'date-released: "{datetime.now().strftime("%Y-%m-%d")}"\n'

    with citation.open("w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
