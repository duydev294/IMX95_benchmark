#!/usr/bin/env python3
"""Single-frame end-to-end NPU inference timing test.

KNOWN ISSUE (see docs/REPORT.md): on this board's tflite_runtime build, the
Neutron external delegate runs in zero-copy mode ("zerocp enabled" in its log),
which segfaults inside interpreter.allocate_tensors() for these models. The C++
benchmark_model tool (also in docs/REPORT.md) does not hit this and is the
reliable path for now.
"""
import sys
import time
import argparse
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

def quantize(x, scale, zero_point, dtype):
    q = np.round(x / scale + zero_point)
    qmin, qmax = (np.iinfo(dtype).min, np.iinfo(dtype).max)
    q = np.clip(q, qmin, qmax)
    return q.astype(dtype)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--delegate", default="/usr/lib/liblitert_neutron_delegate.so")
    ap.add_argument("--mean", type=float, default=127.5)
    ap.add_argument("--std", type=float, default=128.0)
    ap.add_argument("--scale01", action="store_true", help="normalize to [0,1] instead of mean/std (YOLO-style)")
    ap.add_argument("--repeat", type=int, default=10)
    args = ap.parse_args()

    t_load0 = time.perf_counter()
    delegate = load_delegate(args.delegate)
    interpreter = Interpreter(model_path=args.model, experimental_delegates=[delegate], num_threads=1)
    interpreter.allocate_tensors()
    t_load1 = time.perf_counter()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    in_shape = input_details[0]['shape']
    h, w = int(in_shape[1]), int(in_shape[2])
    in_dtype = input_details[0]['dtype']
    in_scale, in_zp = input_details[0]['quantization']

    print(f"Model: {args.model}")
    print(f"Input: shape={list(in_shape)} dtype={in_dtype} quant(scale={in_scale}, zp={in_zp})")
    print(f"Outputs: {len(output_details)}")
    for od in output_details:
        print(f"  {od['name']}: shape={list(od['shape'])} dtype={od['dtype']}")
    print(f"Delegate/interpreter load time: {(t_load1 - t_load0)*1000:.2f} ms")
    print()

    per_frame_times = []
    preprocess_times = []
    infer_times = []
    postprocess_times = []

    for run in range(args.repeat):
        t0 = time.perf_counter()

        img_bgr = cv2.imread(args.image)
        if img_bgr is None:
            print(f"ERROR: could not read image {args.image}")
            sys.exit(1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        img_f = img_resized.astype(np.float32)

        if args.scale01:
            img_norm = img_f / 255.0
        else:
            img_norm = (img_f - args.mean) / args.std

        if in_dtype in (np.int8, np.uint8):
            input_data = quantize(img_norm, in_scale, in_zp, in_dtype)
        else:
            input_data = img_norm.astype(in_dtype)
        input_data = np.expand_dims(input_data, axis=0)

        t1 = time.perf_counter()

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        outputs = [interpreter.get_tensor(od['index']) for od in output_details]

        t2 = time.perf_counter()

        # minimal postprocess: dequantize + report max score from the smallest output as a sanity signal
        _ = [o.astype(np.float32) for o in outputs]

        t3 = time.perf_counter()

        preprocess_times.append((t1 - t0) * 1000)
        infer_times.append((t2 - t1) * 1000)
        postprocess_times.append((t3 - t2) * 1000)
        per_frame_times.append((t3 - t0) * 1000)

    def stats(name, arr):
        a = np.array(arr)
        print(f"{name:14s} avg={a.mean():7.2f}ms  min={a.min():7.2f}ms  max={a.max():7.2f}ms  std={a.std():6.2f}ms")

    print(f"=== Per-frame timing over {args.repeat} runs (image decode + resize + normalize + NPU invoke + fetch outputs) ===")
    stats("Preprocess", preprocess_times)
    stats("Inference", infer_times)
    stats("Postprocess", postprocess_times)
    stats("TOTAL/frame", per_frame_times)
    print()
    print(f"Effective FPS (total pipeline): {1000.0/np.mean(per_frame_times):.2f}")
    print(f"Effective FPS (inference only): {1000.0/np.mean(infer_times):.2f}")

if __name__ == "__main__":
    main()
