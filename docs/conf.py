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
sys.path.insert(0, os.path.abspath('..'))

# Read version without importing the full package (avoids needing ML backends)
_version_globals = {}
with open(os.path.join(os.path.abspath('..'), 'tsgm', 'version.py')) as _f:
    exec(_f.read(), _version_globals)

# Mock heavy ML dependencies that are not available on RTD
autodoc_mock_imports = [
    "tensorflow", "tensorflow_probability", "tf_keras",
    "torch", "torchvision",
    "jax", "jaxlib",
    "keras",
]

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
autoapi_keep_files = True

source_suffix = '.rst'
#master_doc = 'index'

project = 'tsgm'
copyright = '2022, Alexander Nikitin'
author = 'Alexander Nikitin'

# The full version, including alpha/beta/rc tags
release = _version_globals['__version__']

default_role = "any"  # try and turn all `` into links
add_module_names = False  # Remove namespaces from class/method signatures


html_theme = 'sphinx_rtd_theme'

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

html_logo = "_static/logo_docs.png"


# theme-specific options. see theme docs for more info
html_theme_options = {
    'collapse_navigation': False,
    'logo_only': True,
}

# If True, show link to rst source on rendered HTML pages
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]
