from datetime import datetime
from pathlib import Path

from tango.version import VERSION


def main():
    changelog = Path("CHANGELOG.md")

    with changelog.open() as f:
        lines = f.readlines()

    for i in range(len(lines)):
        line = lines[i]
        if line.startswith("## Unreleased"):
            lines.insert(i + 1, "\n")
            lines.insert(
                i + 2,
                f"## [v{VERSION}](https://github.com/allenai/tango/releases/tag/v{VERSION}) - "
                f"{datetime.now().strftime('%Y-%m-%d')}\n",
            )
            break
    else:
        raise RuntimeError("Couldn't find 'Unreleased' section")

    with changelog.open("w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
