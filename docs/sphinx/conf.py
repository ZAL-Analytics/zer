project   = "zer"
copyright = "2026, ZAL Analytics"
author    = "ZAL Analytics"
release   = "1.0.0"

extensions = [
    "sphinx_copybutton",
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
# Copy-button: strip shell prompts                                              #
# --------------------------------------------------------------------------- #

copybutton_prompt_text = r"^\$ |^>>> "
copybutton_prompt_is_regexp = True

# --------------------------------------------------------------------------- #
# Pygments                                                                      #
# --------------------------------------------------------------------------- #

pygments_style       = "monokai"
pygments_dark_style  = "monokai"
