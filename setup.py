from collections import defaultdict
from pathlib import Path
from setuptools import setup, find_packages


def parse_requirements_file(path, allowed_extras: set = None, include_all_extra: bool = True):
    requirements = []
    extras = defaultdict(list)
    with open(path) as requirements_file:
        import re

        def fix_url_dependencies(req: str) -> str:
            """Pip and setuptools disagree about how URL dependencies should be handled."""
            m = re.match(
                r"^(git\+)?(https|ssh)://(git@)?github\.com/([\w-]+)/(?P<name>[\w-]+)\.git", req
            )
            if m is None:
                return req
            else:
                return f"{m.group('name')} @ {req}"

        for line in requirements_file:
            line = line.strip()
            if line.startswith("#") or len(line) <= 0:
                continue
            req, *needed_by = line.split("# needed by:")
            req = fix_url_dependencies(req.strip())
            if needed_by:
                for extra in needed_by[0].strip().split(","):
                    extra = extra.strip()
                    if allowed_extras is not None and extra not in allowed_extras:
                        raise ValueError(f"invalid extra '{extra}' in {path}")
                    extras[extra].append(req)
                if include_all_extra and req not in extras["all"]:
                    extras["all"].append(req)
            else:
                requirements.append(req)
    return requirements, extras


# Find all integrations.
integrations = set(
    p.name
    for p in Path("tango/integrations").glob("*")
    if p.is_dir() and not p.name[0] in {".", "_"}
)

# Load requirements.
install_requirements, extras = parse_requirements_file(
    "requirements.txt", allowed_extras=integrations
)
dev_requirements, dev_extras = parse_requirements_file(
    "dev-requirements.txt", allowed_extras={"examples"}, include_all_extra=False
)
extras["dev"] = dev_requirements
extras.update(dev_extras)

# Validate extras.
assert "all" in extras
assert "dev" in extras
assert "examples" in extras
for integration in integrations:
    assert integration in extras

# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here so we don't import `cached_path` whilst setting up.
VERSION = {}  # type: ignore
with open("tango/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

setup(
    name="ai2-tango",
    version=VERSION["VERSION"],
    description="A library for choreographing your machine learning research.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="",
    url="https://github.com/allenai/tango",
    author="Allen Institute for Artificial Intelligence",
    author_email="contact@allenai.org",
    license="Apache",
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests", "test_fixtures", "test_fixtures.*"],
    ),
    entry_points={"console_scripts": ["tango=tango.__main__:main"]},
    install_requires=install_requirements,
    extras_require=extras,
    python_requires=">=3.7.1",
)
