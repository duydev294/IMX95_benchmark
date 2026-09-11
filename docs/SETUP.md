# Setup

## 1. Target device

- Board: NXP FRDM-IMX95-PRO (chip: i.MX95, NPU: eIQ Neutron)
- OS: NXP i.MX Release Distro (Yocto), kernel 6.18.2, release codename "whinlatter" (built 2026-02-11)
- Access: `ssh root@<device-ip>` (key-based, no password)
- Already installed on-device: `neutron` (firmware/driver), `litert-neutron-delegate`,
  `tensorflow-lite-neutron-delegate`, TFLite 2.19.0 (`/usr/bin/tensorflow-lite-2.19.0/examples/benchmark_model`),
  `tflite_runtime` (Python), ONNX Runtime, OpenCV 4.12.

No extra setup is needed on the device itself — the NPU stack ships with the image.

## 2. Host-side conversion toolchain (Python 3.10, x86_64)

The model conversion tools are NOT on the device; they run on a host PC and produce
a `.tflite` file that gets copied to the device. Install from NXP's eIQ package index:

```bash
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip

# ONNX -> TFLite (quantize + convert)
pip install --index-url https://eiq.nxp.com/repository --extra-index-url https://pypi.org/simple \
  eiq-onnx2tflite

# TFLite -> Neutron-compiled TFLite (target imx95)
pip install --index-url https://eiq.nxp.com/repository --extra-index-url https://pypi.org/simple \
  eiq-neutron-sdk==3.2.2

# ONNX graph simplifier (folds dynamic-shape ops into constants; needed for some
# exports, e.g. models with Shape/Gather/Concat-based reshapes)
pip install onnxsim
```

This installs three CLI tools used by `scripts/convert_and_compile.sh`:
`onnx2quant`, `onnx2tflite` (from `eiq-onnx2tflite`), and `neutron_compiler` (from `eiq-neutron-sdk`).

> Note: `eiq-neutron-sdk` is versioned per NXP BSP release (there's also a smaller
> standalone `neutron-converter-sdk-YY-MM` package on the same index, but it's
> missing native dependencies — use the self-contained `eiq-neutron-sdk` wheel instead).

## 3. Host-side test-frame preprocessing

```bash
pip install pillow numpy
```

Used by `scripts/preprocess_image.py` to turn a JPEG into the raw float32 NCHW
binary blob expected by `benchmark_model --input_layer_value_files`.
