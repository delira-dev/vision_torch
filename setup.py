import os
from setuptools import find_packages, setup


def resolve_requirements(file):
    if not os.path.isfile(file):
        file = os.path.join(os.path.dirname(__file__), "requirements", file)
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(
                    os.path.join(os.path.dirname(file), r.split(" ")[1]))
            else:
                requirements.append(r)
    return requirements


def read_file(file):
    with open(file) as f:
        content = f.read()
    return content


requirements = resolve_requirements(__file__.replace("setup.py",
                                                     "requirements.txt"))
readme = read_file(__file__.replace("setup.py", "README.md"))


setup(
    name='deliravision-torch',
    packages=find_packages(),
    url='https://github.com/delira-dev/vision_torch',
    test_suite="unittest",
    long_description=readme,
    long_description_content_type='text/markdown',
    maintainer="Michael Baumgartner",
    maintainer_email="michael.baumgartner@rwth-aachen.de",
    license='BSD-2',
    install_requires=requirements,
    tests_require=["coverage"],
    python_requires=">=3.5"
)
