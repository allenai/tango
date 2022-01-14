# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

from tango.version import VERSION, VERSION_SHORT  # noqa: E402

# -- Project information -----------------------------------------------------

project = "AI2 Tango"
copyright = "2021, Allen Institute for Artificial Intelligence"
author = "Allen Institute for Artificial Intelligence"
version = VERSION_SHORT
release = VERSION


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx_copybutton",
]

suppress_warnings = ["myst.header"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]

source_suffix = [".rst", ".md"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "datasets": ("https://huggingface.co/docs/datasets", None),
    "transformers": ("https://huggingface.co/docs/transformers", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "pytorch_lightning": ("https://pytorch-lightning.readthedocs.io/en/latest", None),
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

html_title = f"ai2-tango v{VERSION}"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["css/custom.css"]

html_favicon = "_static/favicon.ico"

html_theme_options = {
    "announcement": "ai2-tango is currently under rapid development. Use at your own risk!",
    "light_css_variables": {
        "color-announcement-background": "#1B4596",
        "color-announcement-text": "#FFFFFF",
    },
    "dark_css_variables": {},
    "light_logo": "tango_final_squareish.png",
    "dark_logo": "tango_final_squareish.png",
}

python_use_unqualified_type_names = True
