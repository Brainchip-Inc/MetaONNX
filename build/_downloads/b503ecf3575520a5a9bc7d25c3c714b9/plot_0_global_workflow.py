"""
Global MetaONNX workflow
=========================

This example demonstrates how to deploy an ONNX model to Akida hardware using
the onnx2akida toolkit. Starting from an ONNX model, we'll show how to:

1. Convert and analyze compatibility with Akida hardware
2. Display compatibility reports
3. Create hybrid models combining Akida-compatible and ONNX operators
4. Generate inference models for deployment

We'll use a MobileNetV4 model exported from HuggingFace as our example, though the workflow
applies to any ONNX model.

.. figure:: ../../img/execution_flow.png
   :target: ../../_images/execution_flow.png
   :alt: Overall MetaONNX workflow
   :scale: 25 %
   :align: center

   Global MetaONNX workflow

"""

######################################################################
# 1. Export model to ONNX format
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

######################################################################
# 1.1. Export MobileNetV4 from HuggingFace
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We'll export a MobileNetV4 model from HuggingFace using the Optimum library.
# This demonstrates the typical workflow of obtaining an ONNX model for analysis.
#
# You can also export models from other frameworks:
#
# * `tf2onnx <https://onnxruntime.ai/docs/tutorials/tf-get-started.html>`__ for TensorFlow
# * `torch.onnx <https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html>`__ for PyTorch

import os

from optimum.exporters.onnx import main_export

model_dir = "mbv4"
main_export("timm/mobilenetv4_conv_small.e2400_r224_in1k", output=model_dir)
print(f"Model exported to {model_dir}/model.onnx")


######################################################################
# 1.2. Load the ONNX model
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Load the exported ONNX model for analysis.

import onnx

model_path = os.path.join(model_dir, "model.onnx")
model = onnx.load(model_path)
print(f"\nLoaded ONNX model from {model_path}")
print(f"Model has {len(model.graph.node)} nodes.")

######################################################################
# 2. Convert to Akida
# ~~~~~~~~~~~~~~~~~~~
#

######################################################################
# 2.1. Convert and get compatibility information
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The main entry point is the `convert <../../api_reference/onnx2akida_apis.html#onnx2akida.convert>`__
# function, which analyzes the ONNX model and returns both a `HybridModel
# <../../api_reference/onnx2akida_apis.html#hybridmodel>`__ and detailed compatibility information.
# The `input_shape` parameter specifies the expected input dimensions for the model. Models can be
# exported with a dynamic shape, but quantization and later Akida conversion and mapping need all
# input dimensions to be fixed.

from onnx2akida import convert

# Convert the model and analyze compatibility
# For MobileNetV4, the input shape is (channels, height, width)
print("\nAnalyzing model compatibility with Akida hardware...")
hybrid_model, compatibility_info = convert(model, input_shape=(3, 224, 224))

######################################################################
# 2.2. Display compatibility report
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The obtained `ModelCompatibilityInfo
# <../../api_reference/onnx2akida_apis.html#onnx2akida.compatibility_info.ModelCompatibilityInfo>`__
# object contains detailed information about which nodes and subgraphs are compatible with Akida
# hardware. Use `print_report <../../api_reference/onnx2akida_apis.html#onnx2akida.print_report>`__
# to display a comprehensive analysis.

from onnx2akida import print_report

# Print detailed compatibility report
print_report(compatibility_info, hybrid_model)

######################################################################
# The report shows:
#
# - The list of incompatibles operation types,
# - The list of incompatibilities indexed by node and by stage (quantization, conversion, mapping)
#   indicating where an incompatibility was found and why,
# - Overall compatibility percentage,
# - The memory report for Akida to CPU transfers.

######################################################################
# 2.3. Understanding the HybridModel
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The returned `HybridModel <../../api_reference/onnx2akida_apis.html#hybridmodel>`__ object
# represents a model that can contain both:
#
# * Akida-compatible submodels (will be accelerated on Akida hardware)
# * Standard ONNX operators (will run on CPU via ONNXRuntime)
#
# This hybrid approach allows partial acceleration even when not all operations
# are Akida-compatible.
#
# .. Warning:: Inference is not possible on the `HybridModel` directly. You have to explicitly
#              generate an inference model as shown in the next section.

######################################################################
# 3. Generate inference model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

######################################################################
# 3.1. Generate hybrid inference model with Akida device
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To create a deployable inference model, you need an Akida device.
#
# .. important::
#    A 2.0 FPGA device like available in `Akida Cloud <https://brainchip.com/aclp/>`__ is used here
#    for demonstration.

import akida

# Check for available Akida devices
assert len(devices := akida.devices()) > 0, "No device found, this example needs a 2.0 device."
print(f'Available devices: {[dev.desc for dev in devices]}')

######################################################################
# Inference happens on a device, so we need to map the hybrid model onto it. This can be done using
# `HybridModel.map
# <../../api_reference/onnx2akida_apis.html#onnx2akida.hybrid_model.HybridModel.map>`__ like shown
# below.

# Map on the device
fpga_device = devices[0]
try:
    hybrid_model.map(fpga_device)
except RuntimeError as e:
    print("Mapping failed:\n", e)

######################################################################
# Mapping the HybridModel onto the Akida device after conversion might fail: while some layers are
# supported by Akida hardware, they might not fit on device due to resource constraints.
# In such cases, you can try mapping on a larger virtual device - but that cannot be used for
# inference, it only serves for prototyping - or you can go back to model conversion and provide the
# device as a `convert <../../api_reference/onnx2akida_apis.html#onnx2akida.convert>`__ parameter.

hybrid_model, compatibility_info = convert(model, input_shape=(3, 224, 224), device=fpga_device)

######################################################################
print_report(compatibility_info, hybrid_model)

######################################################################
# The conversion algorithm knows the resource limitations, so it now avoids converting parts
# that do not fit on the device. That is why there are more incompatibilities (the node that was too
# big to fit on 6-node device will run on CPU), but operations that were mapped on the device can be
# accelerated by it.

# Generate the inference model
infer_model = hybrid_model.generate_inference_model()

######################################################################
# 3.2. Save the inference model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Once generated, the inference model can be saved for deployment.

inference_model_path = "model_inference.onnx"
onnx.save(infer_model, inference_model_path)

#####################################################################
# The inference model is a standard ONNX model that can be executed using ONNXRuntime. It's graph
# can be visualized with `Netron <https://netron.app>`__ and it will show ``AkidaOp`` nodes that are
# custom wrappers for all Akida-accelerated submodels. It will also contain ``Transpose`` nodes
# between ONNX and AkidaOp operators are automatically inserted to handle the different data layout
# conventions (NCHW for ONNX, NHWC for Akida).

######################################################################
# 3.3. Perform an inference
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The inference model can be executed using ONNXRuntime and the provided `AkidaInferenceSession
# <../../api_reference/onnx2akida_apis.html#onnx2akida.inference.AkidaInferenceSession>`__.

import numpy as np
from onnx2akida.inference import AkidaInferenceSession

# Generate random input samples with shape (batch_size, channels, height, width)
input_samples = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Prepare and run inference
session = AkidaInferenceSession(inference_model_path)
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_samples})
print(f"Output shape: {outputs[0].shape}")

######################################################################
# 4. Summary
# ~~~~~~~~~~
#
# The onnx2akida workflow enables you to:
#
# 1. **Analyze** any ONNX model for Akida compatibility
# 2. **Identify** which operations can be accelerated on Akida hardware
# 3. **Generate** hybrid models that combine Akida acceleration with standard ONNX operators
# 4. **Deploy** optimized inference models using ONNXRuntime
#
# This approach maximizes hardware acceleration while maintaining full model functionality,
# even when only portions of the model are Akida-compatible.
