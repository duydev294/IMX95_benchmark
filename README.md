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

## Kết quả tóm tắt (đo trên board, ảnh thật)

| Model | Latency/frame (avg) | FPS |
|---|---|---|
| SCRFD-500MF (bnkps) | 15.14 ms | ~66 |
| YOLOv8n-pose | 45.48 ms | ~22 |

Chi tiết đầy đủ (log gốc, dmesg, quá trình debug lỗi compile SCRFD, bug num_threads,
bug segfault Python zero-copy delegate...) xem **[docs/REPORT.md](docs/REPORT.md)**.
