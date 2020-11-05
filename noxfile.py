# noxfile.py
import nox

locations = "delta"


@nox.session(python=["3.8"])
def lint(session):
    args = session.posargs or locations
    session.run("flake8", external=True)
    # session.run("flake8", external=True, *args)

# End of file noxfile.py
