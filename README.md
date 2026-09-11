# i.MX95 Neutron NPU Benchmark

Kiểm tra và benchmark thực tế NPU (eIQ Neutron) trên board **NXP FRDM-IMX95-PRO**
(192.168.100.162, chip i.MX95) — xác nhận môi trường AI chạy được trên NPU thật
(không phải fallback CPU), với 2 model: **SCRFD-500MF** (face detection + 5 keypoints)
và **YOLOv8n-pose** (human pose estimation).

## Cấu trúc repo

```
scripts/
  preprocess_image.py     # JPEG -> raw float32 NCHW binary input
  convert_and_compile.sh  # ONNX -> INT8 quant -> TFLite -> Neutron-compiled TFLite
  run_frame.py            # full-pipeline Python test (đọc ảnh + NPU inference), có bug đã biết — xem docs/REPORT.md
models/
  source/                 # ONNX gốc (điểm bắt đầu, để tái tạo)
  compiled/                # *_neutron.tflite — model cuối cùng, sẵn sàng chạy trên NPU
images/
  face.jpg, person.jpg           # ảnh test gốc
  face_scrfd.bin, person_yolo.bin # input đã tiền xử lý (float32 NCHW 1x3x640x640)
docs/
  SETUP.md    # cài công cụ convert (host) — index eIQ của NXP
  REPORT.md   # báo cáo đầy đủ: nguồn model, pipeline, kết quả benchmark, bug đã gặp
```

## Chạy nhanh trên board

```bash
scp models/compiled/scrfd_500m_kps_neutron.tflite root@<device-ip>:/root/
scp images/face_scrfd.bin root@<device-ip>:/root/

ssh root@<device-ip> '
cd /usr/bin/tensorflow-lite-2.19.0/examples
./benchmark_model --graph=/root/scrfd_500m_kps_neutron.tflite \
  --external_delegate_path=/usr/lib/liblitert_neutron_delegate.so \
  --input_layer=images --input_layer_shape=1,3,640,640 \
  --input_layer_value_files=images:/root/face_scrfd.bin \
  --num_threads=1 --num_runs=10
'
```

⚠️ **Luôn dùng `--num_threads=1`** — `--num_threads=2` gây crash/reboot board tái lặp
(xem `docs/REPORT.md` mục "Bug đã phát hiện").

## Kết quả tóm tắt

| | SCRFD-500MF (bnkps) | YOLOv8n-pose |
|---|---|---|
| Nguồn model | `hpc203/scrfd-opencv` — `scrfd_500m_kps.onnx` (BN, có 5 keypoints) | `Xenova/yolov8n-pose` ONNX |
| Input / Output | 1×3×640×640 → 9 outputs (score/bbox/kps ×3 scale) | 1×3×640×640 → 1×56×8400 |
| Op conversion ratio | 109/143 (76.2%) | 311/384 (81.1%) |
| Neutron ops trong subgraph | 72 | 200 |
| Latency ước tính (compiler, NPU-only) | 3.51 ms @1GHz | 10.24 ms @1GHz |
| Latency đo thực tế trên board (20 runs, `--num_threads=1`) | avg 15.31 ms, std 0.26 ms, min 15.05 / max 16.04 | avg 45.19 ms, std 1.13 ms, min 43.47 / max 47.63 |
| Latency đo với ảnh thật (khác data ngẫu nhiên) | avg 15.14 ms | avg 45.48 ms |
| Model size (compiled) | 954 KB | 3.38 MB |
| FPS tương đương | ~66 | ~22 |

Vì sao latency ước tính và đo thực tế lệch nhau (4–4.4×) — xem
[docs/REPORT.md, mục 9](docs/REPORT.md#9-vì-sao-ước-tính-latency-của-compiler-lệch-với-đo-thực-tế) —
tóm lại: ước tính NPU-only tự nó khá chính xác, phần lệch đến từ op CPU fallback
(chủ yếu Transpose chuyển layout) mà ước tính không tính vào.

Chi tiết đầy đủ (log gốc, dmesg, quá trình debug lỗi compile SCRFD, bug num_threads,
bug segfault Python zero-copy delegate...) xem **[docs/REPORT.md](docs/REPORT.md)**.
