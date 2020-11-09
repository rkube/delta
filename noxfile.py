# noxfile.py
"""nox file for the Delta project.

See nox documentation at https://nox.thea.codes/en/stable/tutorial.html
"""

import nox

locations = "delta"


@nox.session()
def lint(session):
    """Syntax and docstring linting."""
    session.run("flake8")

# End of file noxfile.py
