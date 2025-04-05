# Configuration file for the Sphinx documentation builder.

# -- Project information
from datetime import datetime
from sphinx.application import Sphinx

project = "HCDSIM"
author = "Xikang Feng"
repository_url = "https://github.com/xikanfeng2/HCDSIM"
copyright = f"{datetime.now():%Y}, Xikang Feng."
version = "1.0.0"
release = "1.0.0"

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return "https://somesite/sourcerepo/%s.py" % filename

# -- General configuration
extensions = [
    # read Markdown files
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_autodoc_typehints",  # needs to be after napoleon
    "sphinx.ext.linkcode",
    "sphinx_design",
    "sphinxext.opengraph",
]

html_theme = "sphinx_rtd_theme"
html_title = "HCDSIM"

def setup(app: Sphinx):
    """App setup hook."""
    app.add_config_value(
        "recommonmark_config",
        {
            "auto_toc_tree_section": "Contents",
            "enable_auto_toc_tree": True,
            "enable_math": True,
            "enable_inline_math": False,
            "enable_eval_rst": True,
        },
        True,
    )

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.8", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "markdown_it": ("https://markdown-it-py.readthedocs.io/en/latest", None),
}

nitpick_ignore = [
    ("py:class", name) for name in ["IO", "_io.StringIO", "docutils.nodes.document"]
]
