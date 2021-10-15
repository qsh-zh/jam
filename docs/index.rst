.. raw:: html

        <h1 align="center"> Jammy (Jam) </h1>

        <p align="center">
        <a href="https://pypi.org/project/jammy/">
            <img src="https://img.shields.io/pypi/v/jammy" alt="PyPI" />
        </a>
        <a href="#">
            <img src="https://img.shields.io/pypi/l/jammy" alt="PyPI - License" />
        </a>
        <a href="https://jammy.readthedocs.io/en/stable/index.html">
            <img src="https://img.shields.io/readthedocs/jammy" alt="Read the Docs" />
        </a>
        <a href="#">
            <img src="https://img.shields.io/pypi/pyversions/jammy" alt="PyPI - Python Version" />
        </a>
        <a href="https://github.com/psf/black">
            <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black" />
        </a>
        <p align="center">
            <i>A personal toolbox by <a href="https://qsh-zh.github.io/">Qsh.zh</a>.</i>
        </p>
        </p>



Usage
~~~~~

sample of io
^^^^^^^^^^^^

.. code:: python

    import jammy.io as jio
    from jatorch.utils import as_numpy
    jio.dump("ndarray.npz", np.arange(10))
    jio.dump("foo.pkl", {"foo": as_numpy(torch.arange(10).cuda())})
    ndarray = jio.load("ndarray.npz")
    jio.load("foo.pkl")
    model_dict = jio.load("checkpoint.pth")

.. code:: shell

    $ jam-inspect-file foo.pkl
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
