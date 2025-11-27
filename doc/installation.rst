Installation
============

Supported configurations
------------------------

* **Operating systems:**
    * Any Linux variant compatible with `manylinux 2.28 <https://github.com/pypa/manylinux>`_ (Ubuntu 22.04, Ubuntu 24.04, ...)
* **Python versions:** 3.10 to 3.12
* **cnn2snn version:** 2.18.1
* **onnxruntime version:** 1.19.2

Quick installation
------------------

MetaONNX relies on **onnx2akida** python package that can be setup with
python's pip package manager:

.. code-block:: bash

    pip install onnx2akida=={ONNX2AKIDA_VERSION}

MetaONNX is also built on top of Brainchip usual set of tool included
in `MetaTF <https://doc.brainchipinc.com/>`_, that is QuantizeML, CNN2SNN and Akida. The current
MetaTF version used is {METATF_VERSION}.

.. note::
    We recommend using virtual environment such as `Conda <https://conda.io/docs/>`_.
    Please note that the python version must be explicitly specified when creating a
    conda environment. The specification must be for one of the supported python
    versions listed above.

    Using Conda:

    .. code-block:: bash

      conda create --name onnx2akida_venv python=3.11
      conda activate onnx2akida_venv

    Using python venv:

    .. code-block:: bash

      python3.11 -m venv onnx2akida_venv
      source onnx2akida_venv/bin/activate

Running examples
----------------

The onnx2akida tutorials can be downloaded from the `examples <./examples/index.html>`_
section as python scripts or Jupyter Notebooks. Dependencies needed to replay
the examples can be installed using the :download:`requirements.txt <../requirements.txt>`
file:

.. code-block:: bash

    pip install -r requirements.txt

.. note::
    Please refer to `this link <https://jupyter.org/>`__ for Jupyter Notebook installation
    and configuration.