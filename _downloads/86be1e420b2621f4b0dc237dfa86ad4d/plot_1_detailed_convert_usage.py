"""
Advanced convert usage - Parameters explained by example
========================================================

This tutorial provides a parameter-by-parameter explanation of the
`convert <https://onnx.brainchip.com/api_reference/onnx2akida_apis.html#onnx2akida.convert>`__
function. It explains what each parameter does, when it matters, and how to use it correctly, using
small, practical examples.

The goal is to help users reason about quantization, conversion and device mapping instead of
treating `convert
<https://onnx.brainchip.com/api_reference/onnx2akida_apis.html#onnx2akida.convert>`__ like a black
box.

The tutorial is organized into thematic sections:

    1. Inputs and shapes
    2. Quantization-related parameters
    3. Device and mapping parameters
    4. Common pitfalls and best practices

For this tutorial we will use the
`akidanet_imagenet_224_alpha_1
<https://data.brainchip.com/models/AkidaV2/akidanet/akidanet_imagenet_224_alpha_1.h5>`__
from the MetaTF model zoo, after exporting it to ONNX format.
"""

######################################################################
# 1. Preliminary steps
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# The `convert <https://onnx.brainchip.com/api_reference/onnx2akida_apis.html#onnx2akida.convert>`__
# function performs several distinct stages:
#
#   1. Input shape fixing and model sanitization
#   2. Compatibility checks
#   3. Quantization
#   4. Conversion to a hybrid model
#   5. Optional device-aware mapping
#
# The tutorial will go through the parameters that are listed below.

from onnx2akida import convert
import inspect

print(inspect.signature(convert))

######################################################################
# Load the model

import onnx
import os
import urllib.request

model_filename = "akidanet_imagenet_224_alpha_1.onnx"
model_url = "https://data.brainchip.com/models/AkidaV2/onnx_support/akidanet_imagenet_224_alpha_1.onnx"

_ = urllib.request.urlretrieve(model_url, model_filename)
model = onnx.load(os.path.abspath(model_filename))
print(f"Loaded model with {len(model.graph.node)} nodes")

######################################################################
# The minimal invocation of `convert
# <https://onnx.brainchip.com/api_reference/onnx2akida_apis.html#onnx2akida.convert>`__ only
# requires the model.

from onnx2akida import print_report

hybrid_model, compatibility_info = convert(model)

######################################################################

print_report(hybrid_model, compatibility_info)

######################################################################
# 2. Input-related parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 2.1 input_shape
# ^^^^^^^^^^^^^^^^^^
#
# Having a fully defined non-dynamic model input shape is mandatory for quantization
# and Akida conversion.
# ``input_shape`` parameter can be used to enforce shape on a model that does not fulfill this
# requirement.
# It was possible to convert without it above only because the toy model
# had fixed input dimensions (except the batch dimension which can be dynamic).
# We will use a copy of the model in which we make the input shape dynamic
# to illustrate the importance of this parameter.

clone_model = onnx.ModelProto()
clone_model.CopyFrom(model)

input_tensor = clone_model.graph.input[0]
shape = input_tensor.type.tensor_type.shape.dim


def print_shape(shape):
    print("Input dimensions:")
    for i, d in enumerate(shape):
        if d.dim_param:
            print(f"Dim {i}: {d.dim_param} (Dynamic)")
        else:
            print(f"Dim {i}: {d.dim_value} (Fixed)")


print_shape(shape)

######################################################################
# Let's make the input shape dynamic for demonstration purposes.

for i in range(3):
    shape[i + 1].dim_param = f"dynamic_dim_{i}"

######################################################################
# Verify the change and try converting again.

print_shape(shape)

######################################################################

try:
    hybrid_model, compatibility_info = convert(clone_model)
except RuntimeError as e:
    print(f"Conversion failed as expected: {e}")

######################################################################
# If we provide input_shape (excluding batch), conversion works again.
# Note that convert expects channel-last layout (NHWC) for image data, so we provide the shape
# accordingly.

hybrid_model, compatibility_info = convert(clone_model, input_shape=(224, 224, 3))

######################################################################
# 2.2 input_dtype
# ^^^^^^^^^^^^^^^
#
# ``input_dtype`` defines the expected input type for quantization.
# It is typically either "uint8" (default value) for image data or "int8" for non-image data.
# If the model was trained with a different input range than "uint8", specifying it here
# ensures correct quantization scaling.
# For example, since the model was trained with inputs in the range [0, 255], we can
# try converting with "int8" inputs instead. The percentage of nodes compatible with Akida drops
# to 90.9091% due to the first layers being incompatible with this input range.

hybrid_model, compatibility_info = convert(model, input_dtype="int8")

print_report(hybrid_model, compatibility_info)


######################################################################
# However, if we try converting with "uint16" inputs, the model is 100% compatible.

hybrid_model, compatibility_info = convert(model, input_dtype="uint16")

print_report(hybrid_model, compatibility_info)

######################################################################
# This shows how important it is to use the right dtype to quantize the model, following what was
# done during training.

######################################################################
# DO:
#  - Match input_dtype to the training data range
#
# DON'T:
#  - Assume default input_dtype is always correct

######################################################################
# 2.3 samples and num_samples
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# These parameters are used to provide real calibration data for
# an accurate quantization. If not provided, random data is used for
# calibration. We won't illustrate their use here, as they are covered in detail
# in the `Quantization tutorial
# <https://doc.brainchipinc.com/examples/quantization/plot_0_advanced_quantizeml.html#calibration>`__.
# Quantized model accuracy can be evaluated on the HybridModel, see `Global workflow
# <./plot_0_global_workflow.html#understanding-the-hybridmodel>`__ for more details.
#

######################################################################
# 3. Device and mapping parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Providing a device changes the conversion strategy itself, not just the final mapping.
# When a device is provided, conversion becomes device-aware and avoids generating Akida subgraphs
# that cannot fit. In other words, conversion will also ensure each Akida converted part will also
# map on the given device. If not, the part will be left in ONNX format.

import akida

device = akida.devices()[0]

hybrid_model, compatibility_info = convert(model, device=device)

print_report(hybrid_model, compatibility_info)

######################################################################
# As shown in the report, only 90.9091% of the nodes could be mapped on the device.
# This is because some nodes could not be mapped due to insufficient resources on the
# provided device.
# e.g. "Reason: Cannot map layer
# 'StatefulPartitionedCall/akidanet_1.00_160_1000/conv_2/Conv2D'.
# Not enough hardware components of type CNP1 available.
# 30 are needed but 24 are available."
#
# Note that the two convolutions that could not be mapped due to insufficient resources
# follow one another in the network and they will run consecutively on the
# host processor. Therefore, the two akida models will be formed by the nodes
# before and after these two unmapped nodes.
# If the two nodes were not consecutive, we would need 3 akida models
# to cover the whole network.
#
# It is possible to print the hybrid model summary to visualize the different
# akida models created during conversion and the unmapped nodes which form ONNX sub-models.

hybrid_model.summary()

######################################################################
# It is also possible to use a virtual device which is a simulated version of
# a hardware device. We provide two different devices within
# the akida package:
# `akida.TwoNodesIPv2()
# <https://doc.brainchipinc.com/api_reference/akida_apis.html#akida.TwoNodesIPv2>`__
# and `akida.SixNodesIPv2()
# <https://doc.brainchipinc.com/api_reference/akida_apis.html#akida.SixNodesIPv2>`__ which is
# the software representation of the physical Akida v2 device used when converting the model
# above.
#
# Let's try using the 2 nodes virtual device in
# `convert <https://onnx.brainchip.com/api_reference/onnx2akida_apis.html#onnx2akida.convert>`__.

hybrid_model, compatibility_info = convert(model, device=akida.TwoNodesIPv2())

print_report(hybrid_model, compatibility_info)

######################################################################

hybrid_model.summary()

######################################################################
# With the TwoNodes device, as shown in the report and the summary,
# 56.8182% of the nodes could be mapped and we would need 9 akida models
# to cover the whole network. The reason is that the TwoNodes device
# has much less resources than the SixNodes device and therefore less
# nodes could be mapped.

######################################################################
# DO:
#  - Use device in `convert
#    <https://onnx.brainchip.com/api_reference/onnx2akida_apis.html#onnx2akida.convert>`__ when
#    targeting real hardware
#  - Check the compatibility report to understand resource bottlenecks
#  - Consider that consecutive unmapped nodes will form a single CPU-based segment
#
# DON'T:
#  - Expect 100% node mapping on small virtual devices
#  - Ignore incompatibility reasons - they tell you exactly what resources are missing

######################################################################
# 4. Virtual device optimization computation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The following parameters (enable_hwpr, sram_size, minimal_memory)
# are advanced features intended for fine-tuning the virtual device computation
# when no physical device is provided. Most users should rely on the default
# settings, which work well for typical use cases. Only adjust these if you
# need to optimize for specific memory or resource constraints.
#
# 4.1 enable_hwpr
# ^^^^^^^^^^^^^^^^^
#
# Let us first introduce the concept of Nodes and NPs in Akida devices:
#  - An Akida Node contains a fixed number of 4 NPs (Neural Processors).
#  - An Akida NP can be of type CNP (for convolutional layers), TNP_B (for spatiotemporal TENNs)
#    or FNP (for dense layers).
#  - The total number of NPs in a device is equal to the number of Nodes
#    multiplied by the number of NPs per Node (4 NPs by node).
#  - The term "Node" is also used to refer to a layer in an ONNX graph, which can be confusing.
#    In a device context, "Node" refers to the hardware unit, while in a graph/model context,
#    it refers to a layer.
#
# When no device is provided, convert will return a hybrid model
# that is mapped on a virtual device created during conversion.
# If enable_hwpr is set to False (default), the virtual device assumes that the final device
# must have enough resources to fit each submodel at once.
# Let's inspect the obtained device.

hybrid_model, compatibility_info = convert(model)

print_report(hybrid_model, compatibility_info)

######################################################################


def analyse_hybrid_model(hybrid_model):
    print(f"Number of NPs in final device: {len(hybrid_model.akida_models[0].device.mesh.nps)}")
    print(f"Number of Akida models in the hybrid model: {len(hybrid_model.akida_models)}")
    for i, akida_model in enumerate(hybrid_model.akida_models):
        print(f"\nAkida model {i}:")
        print(f"Number of layers in the Akida model: {len(akida_model.layers)}")
        print(f"Number of sequences in the Akida model: {len(akida_model.sequences)}")
        for j, sequence in enumerate(akida_model.sequences):
            print(f"  - sequence {j}:")
            print(f"    - number of passes: {len(sequence.passes)}")
            for p, passage in enumerate(sequence.passes):
                print(f"      - pass {p} --> {len(passage.layers)} layers")


analyse_hybrid_model(hybrid_model)

######################################################################
# The summary above shows that with the default settings, we end up
# with one akida model containing one pass. The device should have
# at least 208 NPs (52 nodes) to be able to fit the whole submodel
# in one pass.
#
# However, if we enable partial hardware reconfiguration (HWPR) by setting
# enable_hwpr to True, the virtual device created during conversion
# assumes that the device will be partially reconfigured between passes
# as much as possible.
# This generally allows fitting larger submodels on smaller devices
# because resources can be reused across passes.

hybrid_model_hwpr, compatibility_info_hwpr = convert(model, enable_hwpr=True)

print_report(hybrid_model_hwpr, compatibility_info_hwpr)

######################################################################

analyse_hybrid_model(hybrid_model_hwpr)

######################################################################
# As we can see, with enable_hwpr set to True, we end up with the same
# percentage of mapped nodes (100%) and number of akida models,
# but the number of NPs in the device is much smaller (64 NPs, 8 nodes)
# However the akida model now contains 4 passes.
# The cost (clock count) of HWPR is negligible in the overall process,
# so using this parameter will only have an impact on the created
# virtual device (small with components reuse or as big as necessary to
# fit the whole model).

######################################################################
# DO:
#  - Use enable_hwpr=True to allow reusing HW resources and build a smaller device
#  - Expect the final device to be auto-resized to fit each akida model
#  - Remember: 1 Node = 4 NPs
#
# DON'T:
#  - Forget that HWPR allows resource reuse but doesn't eliminate all splits

######################################################################
# 4.2 sram_size
# ^^^^^^^^^^^^^^^^
#
# Used only when ``device`` is None and ``minimal_memory`` is False.
# The sram_size is a tuple that contains the size of shared SRAM
# available inside the mesh for the input_buffer_memory and
# the weight memory.
# The input_buffer_memory is the SRAM needed to store input activations
# (intermediate feature maps) for each neural processing unit (NP).
# When data flows through the network, each layer receives inputs from
# the previous layer - this memory holds those inputs before processing.
# The weight memory is the SRAM needed to store the weights
# (parameters) of the layers assigned to each NP.
# During conversion, if
# `sram_size <https://doc.brainchipinc.com/api_reference/akida_apis.html#akida.NP.SramSize>`__
# is provided, the virtual device created will use this value to determine whether
# a submodel can fit in the available memory. Let's see how to use this parameter.

hybrid_model, compatibility_info = convert(model, sram_size=akida.NP.SramSize(1024, 1024))

print_report(hybrid_model, compatibility_info)

######################################################################
#
# The resulting hybrid models have as expected the same mapping results. However, the device
# created when minimal_memory is enabled has a smaller SRAM size for the input buffer memory and
# the same one for the weight. This is because the biggest layer assigned to the device requires
# less input buffer memory than the default 64 KB. However, the weight memory is being used at its
# maximum (50 KB) so it remains unchanged.

######################################################################
# DO:
#  - Set sram_size to match your target hardware specifications
#  - Understand that smaller SRAM forces more layer splits and akida models
#  - Remember: input_bytes stores activations, weight_bytes stores parameters
#
# DON'T:
#  - Use sram_size when device is provided (it will be ignored)
#  - Set arbitrary values - base them on real hardware capabilities
#  - Forget that 64 KB input + 50 KB weight is the v2 default
#  - Use sram_size with minimal_memory=True (minimal_memory overrides it)

######################################################################
# 4.3 minimal_memory
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Computes the minimal required input buffer and weights memory footprint and overrides
# the default sram_size that is used when sram_size is not explicitly provided, which is
# akida.NP.SramSize_v2 (64 KB for input buffer memory and 50 KB for weight memory).
# When enabled, the virtual device created during conversion assumes that the NPs have just
# enough memory to fit the biggest submodel assigned to them.
# Let's see how this works.

hybrid_model, compatibility_info = convert(model)

######################################################################

print("device input buffer memory: "
      f"{hybrid_model.akida_models[0].device.mesh.np_sram_size.input_bytes}")

print("device weight memory: "
      f"{hybrid_model.akida_models[0].device.mesh.np_sram_size.weight_bytes}")

hybrid_model, compatibility_info = convert(model, minimal_memory=True)

######################################################################

print("device input buffer memory: "
      f"{hybrid_model.akida_models[0].device.mesh.np_sram_size.input_bytes}")

print("device weight memory: "
      f"{hybrid_model.akida_models[0].device.mesh.np_sram_size.weight_bytes}")

######################################################################
#
# The resulting hybrid models have as expected the same mapping results. However,
# the device created when minimal_memory is enabled has a smaller  SRAM size for the input
# buffer memory and the same one for the weight. This is because the biggest submodel assigned
# to an NP requires less input buffer memory than the default 64 KB. However, the weight memory
# is being used at its maximum (50 KB) so it remains unchanged.

######################################################################
# DO:
#  - Use minimal_memory=True to optimize device size for your specific model
#  - Understand that it computes the maximum memory across all NPs
#  - Use it when you want the smallest possible device that fits your model
#  - Check the resulting np_sram_size to understand your model's memory footprint
#
# DON'T:
#  - Combine minimal_memory=True with explicit sram_size (minimal_memory wins)
#  - Expect minimal_memory to reduce the number of akida models
#  - Use it if you need to match specific hardware SRAM specifications
#  - Forget that it's computing per-NP requirements, not total device memory
