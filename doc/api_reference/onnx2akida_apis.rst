
ONNX2AKIDA API
==============

.. automodule:: onnx2akida

    Compatibility
    =============

    ModelCompatibilityInfo
    ----------------------
    .. autoclass:: onnx2akida.compatibility_info.ModelCompatibilityInfo
        :members:

    Hybrid Model Conversion
    -----------------------
    .. autofunction:: onnx2akida.convert

    Compatibility report
    -----------------------
    .. autofunction:: onnx2akida.print_report

    HybridModel
    ===========
    .. autoclass:: onnx2akida.hybrid_model.HybridModel
        :members:
    
    Pipeline
    ========

    Quantization
    ------------
    .. autofunction:: onnx2akida.pipeline.quantize

    Conversion
    ----------
    .. autofunction:: onnx2akida.pipeline.convert_to_hybrid
