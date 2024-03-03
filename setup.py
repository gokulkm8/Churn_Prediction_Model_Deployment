from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."
def get_requirements(filepath):

    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
    
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(name='Credit_Risk_Analysis',
      version='0.0.1',
      description="Machine Learning project to identify loan defaulters",
      author="Gokul Muraleedharan",
      author_email="gokulkm8@gmail.com",
      packages=find_packages(),
      install_requires=get_requirements("requirements.txt")
)