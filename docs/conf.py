# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.append('../signaturescoring')
sys.path.append('../tutorials')
#sys.path.insert(0, os.path.abspath("../tutorials"))
#sys.path.insert(0, os.path.abspath("../signaturescoring"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ANS'
copyright = '2023, ANS contributors'
author = 'ANS contributors'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx.ext.todo",
    "nbsphinx",
    "sphinx_gallery.load_style",
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Extensions --------------------------------------------------------------

# sphinx.ext.todo
# (Support for Todo comments).
todo_include_todos = True

# sphinx.ext.napoleon
# (Support for Google-style docstrings).
napoleon_google_docstring = True
napoleon_use_param = False
napoleon_use_ivar = True


# nbsphinx do not execute jupyter notebooks
nbsphinx_execute = 'never'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# See https://sphinx-themes.org/ for other themes.
# Note that if you change the theme, you may need to modify the GitHub
# Action responsible for deploying the webpage.
html_theme = "pydata_sphinx_theme"

html_static_path = ['_static']

# Options for the theme, as specified there:
# https://pydata-sphinx-theme.readthedocs.io/en/stable/

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/lciernik/ANS_signature_scoring",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        }
    ]
}