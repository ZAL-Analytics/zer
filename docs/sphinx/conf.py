project   = "zer"
copyright = "2026, ZAL Analytics"
author    = "ZAL Analytics"
release   = "1.0.0"

extensions = [
    "sphinx_design",
]

# --------------------------------------------------------------------------- #
# HTML output                                                                   #
# --------------------------------------------------------------------------- #

html_theme          = "sphinxawesome_theme"
html_static_path    = ["_static"]
templates_path      = ["_templates"]
html_css_files      = ["custom.css"]

html_theme_options = {
    "show_prev_next": True,
    "show_scrolltop": True,
}

html_title = "zer an Entity Resolution library for Dutch Administrative Data"

# --------------------------------------------------------------------------- #
# Pygments                                                                      #
# --------------------------------------------------------------------------- #

pygments_style       = "friendly"
pygments_style_dark  = "monokai"
