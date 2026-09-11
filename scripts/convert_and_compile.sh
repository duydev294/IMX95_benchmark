#!/usr/bin/env bash
# Reproduces the ONNX -> INT8 quant -> TFLite -> Neutron-compiled pipeline
# used to produce models/compiled/*.tflite from models/source/*.onnx.
#
# Requires a Python 3.10 venv/target dir with pip packages installed from
# NXP's eIQ index (see docs/SETUP.md for exact install commands):
#   - onnxsim
#   - eiq-onnx2tflite   (index: https://eiq.nxp.com/repository)
#   - eiq-neutron-sdk   (index: https://eiq.nxp.com/repository)
#
# Usage:
#   ./convert_and_compile.sh <input.onnx> <output_neutron.tflite> <input_name> <mode>
#   mode = meanstd | scale01   (only affects nothing here; quantization uses
#          random calibration data, see docs/REPORT.md for accuracy caveats)

set -euo pipefail

ONNX_IN="$1"
NEUTRON_OUT="$2"
INPUT_NAME="${3:-images}"

WORKDIR="$(mktemp -d)"
echo "Working in $WORKDIR"

SIM="$WORKDIR/model_sim.onnx"
QUANT="$WORKDIR/model_quant.onnx"
TFLITE="$WORKDIR/model_quant.tflite"

echo "== 1/4 onnxsim: fold dynamic shapes into constants =="
python3 -m onnxsim "$ONNX_IN" "$SIM" --overwrite-input-shape "${INPUT_NAME}:1,3,640,640"

echo "== 2/4 onnx2quant: INT8 quantize (random calibration data) =="
onnx2quant --use-random-calibration-dataset --per-channel -o "$QUANT" "$SIM"

echo "== 3/4 onnx2tflite: ONNX -> TFLite =="
onnx2tflite -o "$TFLITE" "$QUANT"

echo "== 4/4 neutron-compiler: TFLite -> Neutron (target imx95) =="
neutron_compiler --input "$TFLITE" --output "$NEUTRON_OUT" --target imx95 \
  --convert-inputs-uint8-to-int8 --convert-outputs-uint8-to-int8

echo "Done: $NEUTRON_OUT"
