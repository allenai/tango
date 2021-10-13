"""
Used in CI to create a unique ID for any set of install extras.
"""

import sys


def main():
    extras = sys.argv[1]
    print("-".join(sorted(extras.split(","))))


if __name__ == "__main__":
    main()
