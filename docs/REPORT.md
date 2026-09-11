# Báo cáo kỹ thuật: NPU Neutron trên i.MX95 (FRDM-IMX95-PRO)

Thiết bị: `192.168.100.162`, `root` (SSH key-based), NXP FRDM-IMX95-PRO, chip i.MX95.
OS: NXP i.MX Release Distro, kernel `6.18.2-1.0.0-gf49f45233f7b`, codename **whinlatter**
(build 2026-02-11) — BSP rất mới tại thời điểm test (2026-09-11).

## 1. Xác nhận môi trường NPU hoạt động (bước đầu)

- Device node NPU: `/dev/neutron0`, driver `neutron` load qua remoteproc
  (`4ab00004.imx95-neutron`).
- Firmware: `/usr/lib/firmware/NeutronFirmware.elf`, boot qua remoteproc khi delegate
  được gọi — xác nhận bằng dmesg real-time:
  ```
  remoteproc remoteproc0: powering up neutron-rproc
  remoteproc remoteproc0: Booting fw image NeutronFirmware.elf, size 42332
  remoteproc remoteproc0: remote processor neutron-rproc is now up
  ...
  remoteproc remoteproc0: stopped remote processor neutron-rproc
  ```
- Package sẵn có: `neutron` (2.2.1), `litert-neutron-delegate` (1.2.0),
  `tensorflow-lite-neutron-delegate` (2.19.0), `onnxruntime` (1.23.2), OpenCV (4.12, dnn module).
- Test đầu tiên (mobilenet_v1 quantized **chưa** convert cho Neutron) → delegate nhận
  **0/31 node**, fallback 100% CPU. Kết luận: driver/runtime hoạt động, nhưng cần model
  đã compile riêng bằng **eIQ Neutron Converter** mới chạy được trên NPU thật.

## 2. Toolchain convert (host, không có sẵn trên board)

Cài từ package index chính thức của NXP `https://eiq.nxp.com/repository`:

- `eiq-onnx2tflite` → 2 CLI: `onnx2quant` (INT8 quantize, QDQ), `onnx2tflite` (ONNX → TFLite)
- `eiq-neutron-sdk` (v3.2.2, wheel tự chứa toàn bộ native deps ~188MB) → CLI `neutron_compiler`
  (TFLite → Neutron-compiled TFLite, target `imx95`)

Xem chi tiết lệnh cài trong [SETUP.md](SETUP.md).

Sau khi có mobilenet chạy NPU thành công (avg 1.2ms, ~24 lần nhanh hơn CPU 29ms) để
xác nhận toolchain đúng, tiến hành với 2 model thật theo yêu cầu.

## 3. YOLOv8n-pose

- Nguồn: `Xenova/yolov8n-pose` (HuggingFace, ONNX float32 chuẩn export từ Ultralytics),
  input `1x3x640x640`, output `1x56x8400`.
- Pipeline: `onnx2quant --per-channel --use-random-calibration-dataset` → `onnx2tflite`
  → `neutron_compiler --target imx95`.
- Kết quả compile: **311/384 op (81%)** vào 1 NeutronGraph (200 Neutron op); 2 subgraph nhỏ
  bị đẩy về CPU do compiler chưa hỗ trợ (`Global format selection has failed`,
  `Unknown kernel plugin for operation`). Ước tính latency NPU-only: 10.24 ms.
- Benchmark thực tế (`benchmark_model`, `--num_threads=1`, ảnh người thật `bus.jpg`):
  **avg 45.48 ms**, min 44.02 / max 47.52 ms.

## 4. SCRFD-500MF — hành trình debug lỗi compile

### 4.1 Lỗi ban đầu

Model gốc `500m.onnx` (mirror HuggingFace `RuteNL/SCRFD-face-detection-ONNX`) compile
Neutron **luôn lỗi**:
```
ERROR: neutron-compiler/src/GraphIR/Graph.cpp:3288 New shape should not change the number of elements!
```

### 4.2 Cô lập nguyên nhân (5 bước loại trừ)

1. `onnxsim` (fold shape động thành constant) trước convert → vẫn lỗi.
2. Quantize per-channel vs per-tensor → vẫn lỗi giống hệt.
3. `onnx2tflite --qdq-aware-conversion` vs `--no-qdq-aware-conversion` → vẫn lỗi.
4. Parse trực tiếp file TFLite bằng flatbuffer (`tflite` pip package), kiểm tra toàn bộ
   30 phép Reshape → **không có mismatch số phần tử tĩnh nào** (lỗi không nằm ở model file).
5. Test với model **float32 chưa quantize** → **vẫn lỗi y hệt** → xác nhận lỗi không
   liên quan quantization.

### 4.3 Nguyên nhân gốc

Kiểm tra config gốc trên `github.com/deepinsight/insightface`
(`detection/scrfd/configs/scrfd/scrfd_500m.py`):
```python
bbox_head=dict(
    ...
    feat_channels=64,
    norm_cfg=dict(type='GN', num_groups=16, requires_grad=True),  # GroupNorm!
    ...
)
```
→ **GroupNorm (16 groups, 64 channels/16=4 kênh/group)**. TFLite không có GroupNorm
native, nên `onnx2tflite` decompose thành chuỗi Reshape kiểu
`[1,64,H,W] ↔ [1,16,H·W·4]` — lặp lại 12 lần trong head. Đây chính là pattern
`neutron-compiler` (eiq-neutron-sdk 3.2.2 / SDK 26-03) **chưa xử lý đúng** khi
tracking shape nội bộ (không phải lỗi từ phía model hay pipeline convert).

Config `scrfd_500m_bnkps.py` (cùng repo) dùng:
```python
norm_cfg=dict(type='BN', requires_grad=True)  # BatchNorm — fuse được vào Conv
```
→ "bnkps" = BatchNorm + KeyPoints. BN fuse thẳng vào Conv lúc export ONNX nên
**không còn op normalization riêng, không còn Reshape lạ**.

### 4.4 Giải pháp

Tìm bản ONNX export sẵn của biến thể bnkps: `hpc203/scrfd-opencv` (GitHub) —
`scrfd_500m_kps.onnx`. Kiểm tra xác nhận: 0 node `InstanceNormalization`/`BatchNormalization`
(đã fuse), 191 node (so với 353 của bản GN), 9 output (score/bbox/**kps** × 3 scale).

→ Compile Neutron **thành công ngay lần đầu**: **109/143 op (76.2%)** vào 1 NeutronGraph
(72 Neutron op), ước tính latency NPU-only 3.51 ms.

Benchmark thực tế (`--num_threads=1`, ảnh mặt thật `face.jpg`, 10-20 runs):
**avg 15.14–15.31 ms**, std ~0.26–0.32 ms — rất ổn định.

### Bài học

> Khi chọn model cho eIQ Neutron NPU (SDK hiện tại), **ưu tiên biến thể dùng
> BatchNorm thay vì GroupNorm/InstanceNorm** — BN luôn fuse được vào Conv, cho
> graph đơn giản hơn, tỷ lệ offload NPU cao hơn, và tránh được lỗi compiler với
> pattern Reshape nhóm-kênh.

## 5. Bug ổn định phát hiện: `--num_threads=2` làm crash board

Tái lập **3/3 lần**: chạy `benchmark_model --num_threads=2` trên YOLOv8n-pose (model
có nhiều op CPU fallback chạy song song với 1 node NPU) → board **crash/reboot cứng**
ngay sau khi NPU firmware boot (`remote processor neutron-rproc is now up` là dòng
dmesg cuối cùng trước khi mất kết nối). Không có panic/watchdog log persistent
(rootfs không giữ log qua reboot).

Cô lập bằng thử nghiệm tăng dần:

| num_runs | num_threads | Kết quả |
|---|---|---|
| 1 | 1 | OK |
| 5 | 1 | OK, ổn định |
| 20 | 1 | **OK**, avg 45.19ms, std 1.13ms |
| 20 | 2 | **Crash 3/3 lần** |

→ Không phải do số lần lặp hay rò rỉ bộ nhớ tích lũy — **chính xác là do CPU đa luồng
chạy song song với NPU delegate**. Khả năng cao là race condition/lỗi đồng bộ trong
driver Neutron trên BSP "whinlatter" (rất mới, build 2026-02-11).

**Khuyến nghị: luôn dùng `--num_threads=1`** trên board này cho đến khi có bản vá driver.

## 6. Test với ảnh thật (không phải random data)

`benchmark_model` mặc định tự sinh input ngẫu nhiên. Để đo với **1 frame ảnh thật**:

1. `scripts/preprocess_image.py` — decode JPEG, resize 640×640, normalize
   (SCRFD: `(x-127.5)/128`; YOLOv8: `x/255`), lưu raw `float32` NCHW binary.
2. Feed vào `benchmark_model` qua `--input_layer_value_files=images:<file>.bin`
   (input tensor thực tế của cả 2 model đã compile là `images`, **dtype float32**,
   không phải int8 — model tự có Quantize op nội bộ, I/O ngoài giữ float cho tiện dùng).

Kết quả (ảnh thật) khớp gần như tuyệt đối với benchmark random-data trước đó — hợp lý
vì thời gian NPU chỉ phụ thuộc shape/graph cố định, không phụ thuộc giá trị input:

| Model | Random data | Ảnh thật |
|---|---|---|
| SCRFD-500MF | 15.31 ms | **15.14 ms** |
| YOLOv8n-pose | 45.19 ms | **45.48 ms** |

## 7. Bug đã phát hiện nhưng CHƯA fix: Python `tflite_runtime` segfault

Viết thử `scripts/run_frame.py` (full pipeline: OpenCV đọc ảnh → resize → NPU inference
qua `tflite_runtime.Interpreter` + `load_delegate`) để đo cả preprocessing lẫn inference
trong 1 process Python — **segfault (SIGSEGV) ngay tại `interpreter.allocate_tensors()`**,
đúng lúc NPU firmware boot lên (dmesg và audit log trùng timestamp tuyệt đối). Board
**không** bị crash toàn bộ (chỉ tiến trình `python3` chết) — khác hẳn bug ở mục 5.

Log ngay trước crash luôn dừng ở:
```
INFO: Neutron delegate version: v1.0.0-7399a58e, zerocp enabled.
```
→ Nghi ngờ cao nhất: delegate dùng chế độ **zero-copy** ("zerocp"), yêu cầu buffer
tensor được cấp phát theo cách đặc thù (DMA-mapped) mà API `tflite_runtime` Python
tiêu chuẩn (`set_tensor`/`allocate_tensors`) không tương thích — trong khi công cụ
C++ `benchmark_model` (dùng chung driver/delegate) chạy hoàn toàn ổn định với đúng
model đó. Đây là gợi ý C++ là con đường đáng tin cậy hơn cho ứng dụng thực tế trên
board này, ít nhất cho tới khi xác định rõ cách cấp buffer đúng cho Python binding.

**Chưa debug sâu thêm** (chưa dùng `faulthandler`/gdb để lấy stack trace C-level) —
để dành cho phiên làm việc sau nếu cần pipeline Python đầy đủ.

## 8. Kết luận tổng thể

- NPU Neutron trên i.MX95 **hoạt động thật**, đã benchmark thành công với 2 model
  độc lập (SCRFD-500MF bnkps, YOLOv8n-pose), bằng chứng đầy đủ (log delegate + dmesg
  firmware boot/stop khớp timestamp).
- eIQ Neutron Converter SDK hiện tại (26-03 / eiq-neutron-sdk 3.2.2) **có giới hạn**
  với model dùng GroupNorm/InstanceNorm theo kiểu decompose Reshape — nên tránh,
  ưu tiên biến thể BatchNorm.
- Board có **2 bug ổn định cần lưu ý** khi triển khai thực tế:
  1. `--num_threads=2` (C++) → crash cứng, tái lập 100%.
  2. Python `tflite_runtime` + Neutron delegate zero-copy → segfault tại `allocate_tensors()`.
