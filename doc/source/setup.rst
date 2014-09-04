.. _setup:


*****
Setup
*****

.. _prerequisites:

.. attention::
  Note that `Brainstorm` is the currently proposed name for the library also known as `PyLSTM`.
  This document will be amended to reflect the final name when we decide what it should be.

Prerequisites
=============
First, make sure that all prerequisites are installed. For a typical Debian installation, you can do::

  [sudo] apt-get install cmake python-pip python-dev libopenblas-dev libblas-dev libboost-dev

Install/update numpy, cython and nose. You might want to use a
`virtualenv <http://virtualenv.readthedocs.org/en/latest/virtualenv.html>`_ before doing this
if you do not want to touch your system Python installation::

  [sudo] pip install --upgrade numpy cython nose

.. _installing:

Installing Brainstorm
=====================
Install Brainstorm by making a build directory, entering it and running cmake::

  mkdir build
  cd build
  cmake ..
  make

Run the tests::

  make test
  make pytest

If everything went smoothly, you can install by calling::

  [sudo] ./setup.py install

.. note::

  On some systems (AMD Opteron), you may need to install OpenBLAS from the sources.

