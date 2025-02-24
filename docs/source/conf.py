# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Princess'
copyright = '2025, Carole Périgois'
author = 'Carole Périgois'
release = '0.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "nbsphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
]

html_static_path = ["_static"]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


html_theme = "sphinx_rtd_theme"


autodoc_typehints = "description"  # Pour afficher les types dans la description
autodoc_typehints_format = "short"  # Format court pour les types
autodoc_inherit_docstrings = (
    False  # Ne pas hériter de la docstring des classes parentes
)
add_module_names = False  # Ne pas ajouter les noms de module
napoleon_google_docstring = True  # Pour le style Google
napoleon_include_init_with_doc = False
napoleon_numpy_docstring = True


