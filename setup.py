from setuptools_dso import DSO, setup
from sys import platform

flags: list[str] = []
flag_char: str = "-"
if platform == "win32":
    # MSVC does not support C23 yet, and I can't force setuptools to use MinGW dynamically
    flags.append("/std:c17")
    flag_char = "/"
else:
    flags.append("-std:C23")

flags.append(flag_char+"Wall")

dso = DSO('semi_markov.extension', ['semi_markov/ext/viterbi.c','semi_markov/ext/forward_backward.c'], lang_compile_args={'c':flags})

setup(
    x_dsos = [dso],
    zip_safe = False, # setuptools_dso is not compatible with eggs!
)