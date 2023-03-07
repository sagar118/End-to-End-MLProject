from setuptools import find_packages, setup
from typing import List

REQ_FILE = "requirements.txt"
HYPEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    Read requirements.txt and return a list of all the 
    required packages.

    :param str file_path: Path to requirements.txt
    :return: list of packages
    :rtype: list
    """

    with open(file_path) as handler:
        file = handler.readlines()

    requirements = [req.replace("\n", "") for req in file]
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name = "End-to-End-MLProject",
    version = "0.0.1",
    author = "Sagar",
    author_email = "sgrhacker18@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements(REQ_FILE)
)