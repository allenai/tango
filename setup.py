from setuptools import setup, find_packages

# Load requirements.txt with a special case for allennlp so we can handle
# cross-library integration testing.
with open("requirements.txt") as requirements_file:
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

    install_requirements = []
    for line in requirements_file:
        line = line.strip()
        if line.startswith("#") or len(line) <= 0:
            continue
        install_requirements.append(fix_url_dependencies(line))

# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here so we don't import cached_path whilst setting up.
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
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"],
    ),
    install_requires=install_requirements,
    python_requires=">=3.7.1",
)
