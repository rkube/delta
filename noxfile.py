# noxfile.py
"""nox file for the Delta project.

See nox documentation at https://nox.thea.codes/en/stable/tutorial.html
"""

import nox

locations = "delta"


@nox.session(python=["3.8"])
def lint(session):
    """Syntax and docstring linting."""
    session.run("flake8", external=True)
    session.run("flake8", "--docstring-convention google", external=True)

# End of file noxfile.py
