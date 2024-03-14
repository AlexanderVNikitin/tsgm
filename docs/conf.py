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
import datetime
import sphinx_rtd_theme
import doctest
import tsgm
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    "sphinx.ext.viewcode",  # Add a link to the Python source code for classes, functions etc.
    "sphinx.ext.mathjax",  # Render math via Javascript
    "IPython.sphinxext.ipython_console_highlighting",  # syntax-highlighting ipython interactive sessions
]

### Automatic API doc generation
extensions.append("autoapi.extension")
autoapi_dirs = ["../tsgm"]
autoapi_add_toctree_entry = False
autoapi_python_class_content = "both"
autoapi_options = [
    "members",
    "private-members",
    "special-members",
    "imported-members",
    "show-inheritance",
]

source_suffix = '.rst'
#master_doc = 'index'

project = 'tsgm'
copyright = '2022, Alexander Nikitin'
author = 'Alexander Nikitin'

# The full version, including alpha/beta/rc tags
release = tsgm.__version__

default_role = "any"  # try and turn all `` into links
add_module_names = False  # Remove namespaces from class/method signatures


html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

### intersphinx: Link to other project's documentation (see mapping below)
extensions.append("sphinx.ext.intersphinx")
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
}

autoclass_content = 'both'

### todo: to-do notes
extensions.append("sphinx.ext.todo")
todo_include_todos = True  # pre-1.0, it's worth actually including todos in the docs

### nbsphinx: Integrate Jupyter Notebooks and Sphinx
extensions.append("nbsphinx")
nbsphinx_allow_errors = True  # Continue through Jupyter errors

### sphinxcontrib-bibtex
extensions.append("sphinxcontrib.bibtex")
bibtex_bibfiles = ["refs.bib"]


# Add any paths that contain Jinja2 templates here, relative to this directory.
templates_path = ["_templates"]

# https://sphinxguide.readthedocs.io/en/latest/sphinx_basics/settings.html
# -- Options for LaTeX -----------------------------------------------------
latex_elements = {
    "preamble": r"""
\usepackage{amsmath,amsfonts,amssymb,amsthm}
""",
}

html_theme = 'sphinx_rtd_theme'

html_logo = "_static/logo_docs.png"


# theme-specific options. see theme docs for more info
html_theme_options = {
    "show_prev_next": False,
    "github_url": "https://github.com/",
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
}

# If True, show link to rst source on rendered HTML pages
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]
