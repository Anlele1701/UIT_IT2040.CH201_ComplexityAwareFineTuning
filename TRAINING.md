# Training Commands – MMLU-Pro

---

## Vast.ai + Termius – Kế hoạch chạy trên Cloud GPU

### Tổng quan
Code đã CUDA-ready: `Trainer.__init__` tự động dùng `cuda` nếu có, fallback về `cpu`.
Không cần thay đổi code nào.

### ⚠️ Lessons Learned
- **Tránh instance ở China** — HuggingFace download rất chậm (~1-3 MB/s). Chọn **US/EU**.
- **Set `HF_HOME=/workspace/.cache/huggingface`** trước khi chạy — model cache lưu vào `/workspace` (persistent), không bị mất khi restart. Mặc định `/root/.cache/` nằm trong container layer, mất khi stop.
- Lần đầu chạy mất ~15 phút download Qwen2.5-3B (~6 GB). Lần sau load từ cache, instant.

---

### Bước A – Tạo instance trên Vast.ai

1. Đăng nhập **vast.ai** → **Search** → chọn GPU:
   - Đề xuất: **RTX 4090** (24 GB VRAM, ~$0.3–0.5/h) hoặc **A100 40 GB** (~$1–1.5/h)
   - Filter: **Disk ≥ 40 GB**, **CUDA ≥ 12.1**
2. Template image: chọn **`pytorch/pytorch:2.6.0-cuda12.1-cudnn9-runtime`** (hoặc latest PyTorch)
3. **On-start script** (để trống, setup thủ công)
4. Bật instance → chờ trạng thái **Running**
5. Copy thông tin SSH: `ssh root@<IP> -p <PORT>`

---

### Bước B – Kết nối Termius SSH

1. Mở **Termius** → **New Host**
   - Address: `<IP từ Vast.ai>`
   - Port: `<PORT từ Vast.ai>` (thường không phải 22)
   - Username: `root`
   - Auth: dùng **SSH Key** (upload public key lên Vast.ai trước) hoặc password
2. Kết nối → verify `nvidia-smi` chạy được

---

### Bước C – Upload project qua SFTP (Termius)

Chỉ upload những gì cần thiết cho timing test (bỏ `.venv`, `__pycache__`, `.git`):

```
# Cấu trúc cần upload lên remote ~/complexity-aware-fine-tuning/
complexity-aware-fine-tuning/
├── src/                        ← toàn bộ source package
├── data/
│   └── data_splits/
│       └── entropy_fallback/
│           └── qwen/           ← chỉ cần 4 file này:
│               ├── train_df_easy.tsv
│               ├── train_df_middle.tsv
│               ├── valid_df_combined.tsv
│               └── test_balanced_combined_entr.tsv
├── timing_test.py
├── pyproject.toml
├── uv.lock
└── setup_remote.sh
```

**Trong Termius:** SFTP panel → kéo thả folder từ Finder sang remote, hoặc dùng terminal:
```bash
# Từ máy local (nếu muốn dùng rsync thay SFTP):
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
  --exclude='*.pyc' --exclude='.DS_Store' \
  -e "ssh -p <PORT>" \
  /Users/phunguyen/Code/personal/complexity-aware-fine-tuning/ \
  root@<IP>:~/complexity-aware-fine-tuning/
```

---

### Bước D – Setup môi trường trên remote

```bash
# Trong Termius terminal (đã SSH vào remote)
cd ~/complexity-aware-fine-tuning
bash setup_remote.sh
```

Script sẽ tự động:
1. Kiểm tra `nvidia-smi` và GPU
2. Cài `uv` (nếu chưa có)
3. Tạo `.venv` Python 3.12 + `uv sync` từ `uv.lock`
4. Xác nhận `torch.cuda.is_available() = True`

Expected output cuối:
```
torch.cuda.is_available() = True
GPU: NVIDIA GeForce RTX 4090
VRAM: 24.6 GB
CUDA version: 12.1
```

---

### Bước E – Chạy Timing Test

```bash
cd ~/complexity-aware-fine-tuning
source .venv/bin/activate

# ⚠️ Quan trọng: cache model vào /workspace để không mất khi restart
export HF_HOME=/workspace/.cache/huggingface

# Bước 1: Download model trước (interactive, thấy progress bar rõ)
# Chỉ cần làm 1 lần - sau đó cache lại
huggingface-cli download Qwen/Qwen2.5-3B-Instruct

# Bước 2: Sau khi download xong, chạy timing với nohup
# timing_test.py đã được fix: pre-load model trước, chỉ đo thời gian train
nohup python timing_test.py > timing_test.log 2>&1 &
tail -f timing_test.log
```

Expected output cuối (ví dụ với RTX 4090):
```
===================================================
Timing test: ~Xs cho 25 mini-batches
Toc do: ~Y s/mini-batch

Uoc tinh thoi gian thuc:
  Pipeline pt1 Qwen  (...):  Xh
  ...
```

Nếu muốn chạy trong nohup (để không bị ngắt khi đóng terminal):
```bash
nohup python timing_test.py > timing_test.log 2>&1 &
tail -f timing_test.log
```

---

### Bước F – Tắt instance (quan trọng để tiết kiệm tiền)

Sau khi lấy kết quả: **Vast.ai dashboard → Stop instance**.
Lưu kết quả `timing_test.log` về máy local trước khi tắt.

---

---

## Setup (local)

```bash
cd /Users/phunguyen/Code/personal/complexity-aware-fine-tuning
source .venv/bin/activate
```

---

## Bước 1 – Timing Test (chạy trước để ước tính thời gian thực)

Script nhỏ: 200 samples, 1 epoch, không eval → đo thời gian/step rồi suy ra tổng.

```bash
python - << 'EOF'
import time
from pathlib import Path
from reasoning_fine_tune.training.sft.config import TrainConfig
from reasoning_fine_tune.training.sft.train import Trainer

cfg = TrainConfig.preset("Qwen/Qwen2.5-3B-Instruct")
cfg.epochs = 1
cfg.debug = False
cfg.train_sample_size = 200
cfg.val_sample_size = 50
cfg.test_sample_size = 50
cfg.run_eval_on_start = False
cfg.eval_validation_period = 0
cfg.eval_test_period = 0
cfg.save_dir = "/tmp/timing_test"

cfg.train_path = [
    str(Path("data/data_splits/entropy_fallback/qwen/train_df_easy.tsv")),
    str(Path("data/data_splits/entropy_fallback/qwen/train_df_middle.tsv")),
]
cfg.valid_path = str(Path("data/data_splits/entropy_fallback/qwen/valid_df_combined.tsv"))
cfg.test_path = str(Path("data/data_splits/entropy_fallback/qwen/test_balanced_combined_entr.tsv"))
cfg.test_balanced_path = cfg.test_path

t0 = time.time()
trainer = Trainer(cfg)
trainer.train(save=False)
elapsed = time.time() - t0

# --- Extrapolation ---
# pt1: 3000 samples, 10 epochs, batch=8 → ceil(3000/8)*10 = 3750 mini-batches
# pt2: 3000 samples, 10 epochs, batch=2 → ceil(3000/2)*10 = 15000 mini-batches
# SFT: 3000 samples, 20 epochs, batch=8 → 7500 mini-batches
# Full Distill: 3000 samples, 20 epochs, batch=2 → 30000 mini-batches (+ CoT sequences dài ~3-5x)
steps_in_test = 200 / 8  # mini-batches chạy trong test
secs_per_step = elapsed / steps_in_test
print(f"\n{'='*50}")
print(f"Timing test: {elapsed:.1f}s cho {steps_in_test:.0f} steps")
print(f"Tốc độ: {secs_per_step:.2f}s/step")
print(f"\nƯớc tính thời gian (giây/phút/giờ):")
for name, n_steps in [
    ("Pipeline pt1 (x2 models)", 3750 * 2),
    ("Pipeline pt2 (x2 models, CoT ~3x)", 15000 * 2 * 3),
    ("SFT Baseline (x2 models)", 7500 * 2),
    ("Full Distill (x2 models, CoT ~3x)", 30000 * 2 * 3),
]:
    secs = secs_per_step * n_steps
    print(f"  {name}: {secs/3600:.1f}h")
print('='*50)
EOF
```

---

## Bước 2 – MMLU-Pro Training (chạy tuần tự)

> **Lưu ý**: Chỉ chạy 1 job tại 1 thời điểm. pt2 phải chờ pt1 xong mới chạy được.

### 2a. Pipeline – Qwen2.5-3B (~pt1: 6h + pt2: 17h)

```bash
# pt1 và pt2 chạy nối tiếp tự động
python src/experiments/pipeline/pipeline/pipeline_pt1_qwen3b.py && \
python src/experiments/pipeline/pipeline/pipeline_pt2_qwen3b.py
```

### 2b. Pipeline – Phi4-mini (~pt1: 6h + pt2: 17h)

```bash
python src/experiments/pipeline/pipeline/pipeline_pt1_phi4mini.py && \
python src/experiments/pipeline/pipeline/pipeline_pt2_phi4mini.py
```

### 2c. SFT Baseline – Qwen2.5-3B (~10-12h)

```bash
python src/experiments/pipeline/sft_baseline/sft_baseline_qwen3b.py
```

### 2d. SFT Baseline – Phi4-mini (~10-12h)

```bash
python src/experiments/pipeline/sft_baseline/sft_baseline_phi4mini.py
```

### 2e. Full Distillation – Qwen2.5-3B (~45-55h) ⚠️ Khuyên dùng cloud

```bash
python src/experiments/pipeline/full_distill/full_distill_baseline_qwen3b.py
```

### 2f. Full Distillation – Phi4-mini (~45-55h) ⚠️ Khuyên dùng cloud

```bash
python src/experiments/pipeline/full_distill/full_distill_baseline_phi4mini.py
```

---

## Thông số training (từ config)

| Param | pt1 | pt2 | SFT | Full Distill |
|---|---|---|---|---|
| `train_sample_size` | 3000 | 3000 | 3000 | 3000 |
| `batch_size` | 8 | 2 | 8 | 2 |
| `gradient_accumulation` | 32 | 32 | 32 | 32 |
| effective batch | 256 | 64 | 256 | 64 |
| `epochs` | 10 | 10 | 20 | 20 |
| `use_cot` | False | **True** | False | **True** |

---

## Kết quả lưu ở đâu

```
artifacts/pipeline_20epochs/
├── pipeline/
│   ├── qwen/pt1/    training_<PID>_results.jsonl   ← accuracy theo epoch
│   ├── qwen/pt2/    training_<PID>_results.jsonl
│   ├── phi4mini/pt1/
│   └── phi4mini/pt2/
├── sft_baseline/
│   ├── qwen/        training_<PID>_results.jsonl
│   └── phi4mini/
└── full_distill_baseline/
    ├── qwen/
    └── phi4mini/
```

---

## Nếu training bị interrupt

Không có auto-resume — phải chạy lại từ đầu. Nếu muốn giảm rủi ro trên Mac, chạy trong `nohup`:

```bash
nohup python src/experiments/pipeline/pipeline/pipeline_pt1_qwen3b.py > logs/pt1_qwen.log 2>&1 &
tail -f logs/pt1_qwen.log
```
