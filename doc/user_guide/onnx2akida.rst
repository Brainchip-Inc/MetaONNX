onnx2akida toolkit
==================

Overview
--------

The **onnx2akida** package provides tools to evaluate the compatibility of an ONNX model with
Akida hardware. The tool quantizes ONNX models, attempts conversion to Akida format, and reports
which nodes are hardware-compatible. It can also generate **hybrid models** that combine both 
Akida-compatible submodels and standard ONNX operators, enabling inference through the Akida
runtime together with ONNXRuntime.

onnx2akida workflow
-------------------

The typical workflow begins with an ONNX model. If you have a model in another framework,
you can export it to the ONNX format using various tools:

* `tf2onnx <https://onnxruntime.ai/docs/tutorials/tf-get-started.html>`__ for TensorFlow models.
* `torch.onnx <https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html>`__ for PyTorch models.
* `optimum <https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model>`__ for HuggingFace models.

For example, to export a MobileNetV4 model from HuggingFace to ONNX format:

.. code-block:: bash

   pip install optimum timm

.. code-block:: python

   from optimum.exporters.onnx import main_export
   main_export("timm/mobilenetv4_conv_small.e2400_r224_in1k", output="mbv4")

Once the model is available in ONNX format, it can be converted using the onnx2akida main entry
point `convert <../api_reference/onnx2akida_apis.html#onnx2akida.convert>`__  function:

.. code-block:: python

   import onnx
   from onnx2akida import convert

   model = onnx.load("mbv4/model.onnx")
   hybrid_model, compatibility_info = convert(model, input_shape=(3, 224, 224))

The returned `compatibility_info <../api_reference/onnx2akida_apis.html#onnx2akida.compatibility_info.ModelCompatibilityInfo>`__ 
object contains detailed information about which parts of the model are compatible with Akida. Its content can be displayed using:

.. code-block:: python

   from onnx2akida import print_report
   print_report(compatibility_info, hybrid_model)

The `HybridModel <../api_reference/onnx2akida_apis.html#onnx2akida.hybrid_model.HybridModel>`__ 
object can then be used to generate a hybrid inference model containing both Akida submodels and standard 
ONNX operators. Each Akida submodel is represented as an ``AkidaOp`` node in the final graph,
effectively building into a fully ONNX node based model.

.. important::
   To generate the inference model, the conversion must be performed with a physical Akida device.

.. code-block:: python
   
   
   import akida

   # Ensure there is an Akida device available
   assert len(akida.devices()) > 0, "No Akida device found !"
   
   hybrid_model, _ = convert(model, input_shape=(3, 224, 224), device=akida.devices()[0])
   infer_model = hybrid_model.generate_inference_model(dirpath=".")

The ``dirpath`` argument specifies the folder where Akida submodel binaries will be stored.
The resulting hybrid ONNX model is compatible with ONNXRuntime with AkidaExecutionProvider for inference.

.. note::
    The global workflow described here is detailed in the dedicated examples available in
    `onnx2akida examples <../examples/>`_ section.

Command line interface
----------------------

The toolkit also provides command-line utilities for quick prototyping or users who prefer
working outside the Python API.

onnx2akida CLI
~~~~~~~~~~~~~~

To generate a compatibility report for an ONNX model, the following command can be used:

.. code-block:: bash

    onnx2akida -m model.onnx -s model_tagged.onnx

This will show model compatibility percentage with Akida hardware as well as which nodes or 
sequences were not compatible with reasons.
The file ``model_tagged.onnx`` is intentionally not a valid ONNX model.  
Its purpose is to allow visualization of compatibility annotations in the Netron application
`Netron <https://netron.app>`__.

If a model uses dynamic input shapes, the input shape must be specified:

.. code-block:: bash

    onnx2akida -m model.onnx -s model_tagged.onnx --input_shape 3,224,224

onnx2akida-device CLI
~~~~~~~~~~~~~~~~~~~~~

To estimate the minimum Akida device configuration required for an ONNX model:

.. code-block:: bash

    onnx2akida-device -m model.onnx

A more detailed explanation on Device and Mapping will be covered in an upcoming tutorial.
