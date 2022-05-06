.. raw:: html

        <h1 align="center"> Jammy (Jam)</h1>
        <p align="center">
        <a href="https://pypi.org/project/jammy/">
            <img src="https://img.shields.io/pypi/v/jammy?style=for-the-badge" alt="PyPI" />
        </a>
        <a href="#">
            <img src="https://img.shields.io/pypi/l/jammy?style=for-the-badge" alt="PyPI - License" />
        </a>
        <a href="https://github.com/qsh-zh/jam">
            <img src="https://img.shields.io/badge/-github-grey?style=for-the-badge&logo=github" alt="GitHub code" />
        </a>
        <a href="https://gitlab.com/qsh.zh/jam">
            <img src="https://img.shields.io/badge/-gitlab-grey?style=for-the-badge&logo=gitlab" alt="GitLab code" />
        </a>
        <a href="https://jammy.readthedocs.io/en/stable/index.html">
            <img src="https://img.shields.io/readthedocs/jammy?style=for-the-badge" alt="Read the Docs" />
        </a>
        <a href="#">
            <img src="https://img.shields.io/pypi/pyversions/jammy?style=for-the-badge" alt="PyPI - Python Version" />
        </a>
        <a href="https://github.com/psf/black">
            <img src="https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge" alt="Code style: black" />
        </a>
        <p align="center">
            <i>A personal toolbox by <a href="https://qsh-zh.github.io/">Qsh.zh</a>.</i>
        </p>
        </p>



Usage
~~~~~

setup
^^^^^

-  For core package, run ``pip install jammy``
-  To access functions in ``bin``

   .. code:: shell

       git clone https://gitlab.com/qsh.zh/jam.git --recursive
       export PATH=<path_to_jam>/bin:$PATH
       # run python program
       jam-run main.py
       jam-crun 1 main.py # use second nvidia gpu


sample of io
^^^^^^^^^^^^

.. code:: python

    import jammy.io as jio
    from jamtorch.utils import as_numpy
    jio.dump("ndarray.npz", np.arange(10))
    jio.dump("foo.pkl", {"foo": as_numpy(torch.arange(10).cuda())})
    ndarray = jio.load("ndarray.npz")
    jio.load("foo.pkl")
    model_dict = jio.load("checkpoint.pth")

.. code:: shell

    $ jinspect-file foo.pkl
    > python3
    [ins] print(f1)
    # content of foo.pkl

Advanced Usage
~~~~~~~~~~~~~~

-  `A DDP pytorch training
   framework <https://jammy.readthedocs.io/en/stable/jamtorch.ddp.html>`__
-  `Registry <https://jammy.readthedocs.io/en/stable/jammy.utils.html?highlight=registry#jammy.utils.registry.CallbackRegistry>`__
-  MORE TODO


Etymology
^^^^^^^^^


* The naming is inspired from `Jyutping <https://en.wikipedia.org/wiki/Jyutping>`_ of `Qin <https://en.wiktionary.org/wiki/%E6%AC%BD>`_.



MICS
----


* The package and framework are inspired from `Jacinle <https://github.com/vacancy/Jacinle>`_ by `vacancy <https://github.com/vacancy>`_\ , from which I learn and take utility functions shamelessly.


Package Reference
=================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   jammy
   jamnp
   jamtorch
   jamviz
   jamweb


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
