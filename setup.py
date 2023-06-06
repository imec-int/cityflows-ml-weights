import os
from setuptools import find_packages, setup

install_requires = [
    'numpy',
    'pandas',
    'pytest'
]

setup_requirements = [
    'pytest-runner',
    'better-setuptools-git-version'
]

test_requirements = [
    'pytest',
    'nbformat'
]

setup(
    author='Julien Verplanken',
    author_email="Name@equinor.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        'License :: OSI Approved :: MIT License',
    ],

    name="ai-cityflows-straatvinken",
    # version="0.0.1",
    version_config={
      "version_format": "{tag}.dev{sha}",
      "starting_version": "0.0.1"
    },
    description="Predicting traffic intensity from geospatial attributes",
    long_description=open('README.md').read(),
    packages=find_packages('src'),
    package_dir={'': 'src'},
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    install_requires=install_requires
)
