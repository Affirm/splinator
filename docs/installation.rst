Installation
============

Requirements
------------

* Python 3.7.1 or later
* NumPy >= 1.19.0
* SciPy >= 1.6.0
* scikit-learn >= 1.0.0
* pandas >= 1.3.0

Install from PyPI
-----------------

The easiest way to install splinator is via pip::

    pip install splinator

Install from Source
-------------------

To install from source, clone the repository and install using pip::

    git clone https://github.com/yourusername/splinator.git
    cd splinator
    pip install -e .

Development Installation
------------------------

For development, install with the extra dependencies::

    pip install -e ".[dev]"

This will install additional packages needed for development:

* pytest for testing
* matplotlib for plotting
* mypy for type checking
* jupyter for notebooks

Verify Installation
-------------------

You can verify the installation by importing splinator::

    python -c "import splinator; print(splinator.__version__)"

This should print the version number without any errors. 