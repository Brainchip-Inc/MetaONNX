Onnx2akida toolkit
==================

Overview
--------

The **onnx2akida** package provides tools to evaluate the compatibility of an ONNX model with
Akida hardware. It can also generate **hybrid models** that combine both Akida-compatible
submodels and standard ONNX operators, enabling inference through the Akida runtime together
with ONNXRuntime.

Onnx2akida workflow
-------------------

The typical workflow begins with an ONNX model. If you have a model in another framework,
you can export it to the ONNX format using various tools:

1. `tf2onnx <https://onnxruntime.ai/docs/tutorials/tf-get-started.html>`__ for TensorFlow models.
2. `torch.onnx <https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html>`__ for PyTorch models.
3. `optimum <https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model>`__ for HuggingFace models.

For example, to export a MobileNetV3 model from HuggingFace to ONNX format:

.. code-block:: python

   pip install optimum timm

.. code-block:: python

   from optimum.exporters.onnx import main_export
   main_export("timm/mobilenetv3_small_100.lamb_in1k", output="mbv3")

Once the model is available in ONNX format, it can be converted using the onnx2akida
`convert <../api_reference/onnx2akida_apis.html#onnx2akida.convert>`__  function:

.. code-block:: python

   import onnx
   from onnx2akida import convert

   model = onnx.load("mbv3/model.onnx")
   hybrid_model, compatibility_info = convert(model, input_shape=(3, 224, 224))

The returned `compatibility_info <../api_reference/onnx2akida_apis.html#onnx2akida.compatibility_info.ModelCompatibilityInfo>`__ 
object contains detailed information about which parts of the model are compatible with Akida. Its content can be displayed using:

.. code-block:: python

   from onnx2akida import print_report
   print_report(compatibility_info, hybrid_model)

The `HybridModel <../api_reference/onnx2akida_apis.html#onnx2akida.hybrid_model.HybridModel>`__ 
object can then generate a hybrid inference model containing both Akida submodels and standard 
ONNX operators. Each Akida submodel is represented as an ``AkidaOp`` node in the final graph.

.. code-block:: python

   infer_model = hybrid_model.generate_inference_model(dirpath=".")

The ``dirpath`` argument specifies the folder where Akida submodel binaries will be stored.
The resulting hybrid ONNX model is compatible with **onnxruntime-akida** for hybrid inference.

 
Command line interface
----------------------

The toolkit also provides command-line utilities for users who prefer working outside the
Python API.

onnx2akida CLI
~~~~~~~~~~~~~~

To generate a compatibility report for an ONNX model, the following command can be used:

.. code-block:: bash

    onnx2akida -m model_onnx.onnx -s model_tagged.onnx

The file ``model_tagged.onnx`` is intentionally not a valid ONNX model.  
Its purpose is to allow visualization of compatibility annotations in the Netron application
`Netron <https://netron.app>`__.

If a model uses dynamic input shapes, the input shape must be specified:

.. code-block:: bash

    onnx2akida -m model_onnx.onnx -s model_tagged.onnx --input_shape 3,224,224

onnx2akida-device CLI
~~~~~~~~~~~~~~~~~~~~~

To estimate the minimum Akida device configuration required for an ONNX model:

.. code-block:: bash

    onnx2akida-device -m model_onnx.onnx

You can also compute mininum device with hardware partial reconfiguration:

.. code-block:: bash

    onnx2akida-device -m model_onnx.onnx -hwpr
