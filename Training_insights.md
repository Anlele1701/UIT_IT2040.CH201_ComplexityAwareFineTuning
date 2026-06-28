# Training Insights – Pipeline Qwen2.5-3B

## Setup

| | |
|---|---|
| GPU (test run) | NVIDIA GeForce RTX 3090, 24 GB VRAM |
| GPU (actual training) | **A100 PCIE 40 GB** – Japan 🇯🇵 ($0.544/h, DLPerf 95.4) |
| CUDA | 13.0 (RTX 3090) / 13.2 (A100) |
| Model | Qwen/Qwen2.5-3B-Instruct (3.2B params) |
| Platform | Vast.ai |

> Chuyển từ RTX 3090 (China 🇨🇳) → A100 PCIE (Japan 🇯🇵) vì:
> - HuggingFace download từ China cực chậm (4–6 MB/s)
> - A100 có 40 GB VRAM (rộng hơn, tránh OOM cho CoT sequences dài)
> - DLPerf/$ cao nhất trong các option khảo sát (175 DLP/$/hr)

---

## Dataset – Entropy Fallback / Qwen splits

### Kích thước các tập

| File | Rows | Dùng cho |
|---|---|---|
| `train_df_easy.tsv` | **900** | pt1 training |
| `train_df_middle.tsv` | **900** | pt1 training |
| `train_df_hard.tsv` | **900** | pt2 training (CoT) |
| `valid_df_combined.tsv` | **300** | Validation (monitor) |
| `test_balanced_combined_entr.tsv` | **906** | Test (evaluation) |

pt1 tổng train: 900 + 900 = **1800 rows** → 1800/8 = **225 batches/epoch**

### Kiểm tra overlap (data leakage)

| | Train (all) | Valid | Test |
|---|---|---|---|
| Train (all) | — | **29 rows trùng** ⚠️ | 0 ✅ |
| Valid | — | — | 0 ✅ |

- **Train ∩ Test = 0** → test set hoàn toàn tách biệt ✅
- **Valid ∩ Test = 0** → không bị leakage khi report final accuracy ✅
- **Train ∩ Valid = 29 rows** ⚠️ — 29/300 = ~10% valid set có câu hỏi trùng với train. Validation dùng để monitor loss curve (không dùng để model selection) nên ảnh hưởng nhỏ, nhưng cần ghi chú khi báo cáo.

### CoT column

- `distill_response` trong `train_df_hard.tsv`: **900/900 non-null** ✅ — pt2 có đủ CoT labels để train.

---

## Fine-tuning Method: LoRA (SFT)

- **Chỉ train 3.74% parameters** (`trainable: 119,734,272 / total: 3,205,672,960`)
- LoRA target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Config: `r=64, alpha=128, dropout=0.05`
- Full model frozen → chỉ LoRA adapter được cập nhật → tiết kiệm memory và tốc độ

## Memory (RTX 3090 – 24 GB)

- VRAM peak: **~23.4 GB / 24 GB** (sát ngưỡng)
- Transformers tự cấp phát 90% memory cho model, 10% buffer tránh OOM
- `torch_dtype=bfloat16` → giảm memory xuống một nửa so với float32

---

## Training Speed (batch_size=8, non-CoT, RTX 3090)

| Metric | Value |
|---|---|
| Tốc độ thực (tqdm) | **2.35s / mini-batch** |
| Tổng 25 batches (incl. overhead) | 80.6s |
| Overhead (data prep + save) | ~22s |
| Batch shape | `torch.Size([8, 240])` — sequence length ~240 tokens |
| Model load từ cache | ~6 giây |

> ⚠️ Số 3.22s/mini-batch trong output script bao gồm overhead. Số thực GPU là **2.35s/it** từ tqdm.

---

## Ước tính thời gian training (2.35s/step, batch_size=8, RTX 3090)

| Job | Config | Steps | Ước tính |
|---|---|---|---|
| Pipeline pt1 Qwen | 1800s, 5ep, b=8 | 1125 | **~0.7h** |
| Pipeline pt1 Qwen | 1800s, 10ep, b=8 | 2250 | **~1.5h** |
| Pipeline pt2 Qwen | 900s, 5ep, b=2, CoT | ~6750 | **~4.4h** |
| Pipeline pt2 Qwen | 900s, 10ep, b=2, CoT | ~13500 | **~8.8h** |

> ⚠️ pt2 CoT: sequence dài hơn non-CoT ~3-5x → thực tế có thể chậm hơn/step. Cần timing test riêng.

---

## Accuracy & Loss (1 epoch, 200 samples, RTX 3090)

- **Loss**: 2.921
- **Accuracy**: 33.12% sau 1 epoch (random baseline = 10% với 10 lựa chọn)
- Model đã hiểu instruction format `[[number]]` ngay từ pretrained

---

## Gotchas & Fixes

| Vấn đề | Fix |
|---|---|
| `eval_batch_size=1` (default) → eval 906 samples mất ~10 phút | Set `eval_batch_size=8` trong scripts → ~72 giây |
| pt1 có 6 lần eval (1 start + 5 epochs) × 10 phút = 1 giờ lãng phí | Fixed bằng `eval_batch_size=8` |
| HuggingFace download chậm từ China | Dùng instance US/EU |
| `tail -f` không thấy real-time log | Set `PYTHONUNBUFFERED=1` + bỏ `tee` bên trong script |
| Kill shell script không kill Python child | Dùng `pkill -9 -f python3` |
| tmux conflict với Termius | `touch ~/.no_auto_tmux` rồi reconnect |
| CUDA OOM do memory fragmentation (A100 40GB) | Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Vast.ai hết credit → process bị kill | pt1 model vẫn saved → chỉ cần rerun pt2, không cần rerun pt1 |
| pt2 load pt1 model (đã có LoRA) rồi apply LoRA lần 2 | ⚠️ PEFT warning "multiple adapters" — behavior của tác giả, không phải bug. Tác giả không merge LoRA trước khi save pt1 → pt2 stack LoRA thứ 2 lên. Cần báo cáo thầy. |

---

## Discrepancies – Paper vs Code Implementation

| | Paper (Table 11) | Code (thực tế) | |
|---|---|---|---|
| pt1 effective batch | 64 | **256** (batch=8 × grad_accum=32) | ❌ |
| pt2 effective batch | 64 | **64** (batch=2 × grad_accum=32) | ✅ |
| max_new_tokens (pt1/SFT) | 30 | 30 | ✅ |
| max_new_tokens (pt2/Distill) | **8192** | **30** (refactored, bug!) | ❌ |

> ⚠️ **Ảnh hưởng đến kết quả pt2:** Chúng ta fix bằng `max_new_tokens=2048` (thay vì paper's 8192) vì 8192 quá chậm. CoT responses có p99=2390 tokens → ~1% samples bị cắt với 2048. Accuracy pt2 có thể thấp hơn paper một chút do một số CoT chains bị truncated trước khi reach `[[number]]`. Kết quả reported (43.93%) có thể underestimate so với nếu dùng 8192.

**Nguồn gốc discrepancy:**
- `gradient_accumulation=32` là default trong config, **không được override** trong script pt1
- `max_new_tokens` bị đổi từ `12888 → 30` trong commit refactor (7 months ago, danielvyazh9ev) — broke CoT eval
- Fix của chúng ta: set `max_new_tokens=2048` trong `pipeline_qwen_pt2_5ep.py` (đủ cho ~99% CoT responses)

**Cần báo cáo thầy:**
1. pt1 effective batch size = 256 (không phải 64 như paper)
2. max_new_tokens cho CoT: paper=8192, code=30 (bug), fix=2048
3. pt2 stacks LoRA adapter (double LoRA warning) — không merge pt1 LoRA trước

**Cấu trúc thực tế:**
```
Base Qwen → LoRA_pt1 (frozen, từ pt1 save) → LoRA_pt2 (trainable)
```

**Cấu trúc lý tưởng:**
```
Base Qwen + merged(LoRA_pt1) → LoRA_pt2 (trainable)
```

**Observation:** `trainable params: 119,734,272 (3.7351%)` — giống hệt pt1 → chỉ LoRA_pt2 được train, LoRA_pt1 bị frozen.

**Rủi ro:**
- LoRA_pt1 không được merge → overhead memory nhỏ
- Gradient đi qua 2 adapter layers → training không tối ưu
- Knowledge từ pt1 **vẫn ảnh hưởng** forward pass (vì LoRA_pt1 vẫn trong model) → không mất signal

**Kết luận:** Vì đây là code gốc tác giả và họ đã có kết quả → không sửa. Ghi chú khi report:
> *"pt2 applies a second LoRA adapter on top of the saved pt1 model without merging pt1's adapter first. This may slightly reduce efficiency compared to a merge-then-finetune approach."*

---

## Lessons Learned (Vast.ai setup)

- **Tránh instance China** — HuggingFace download rất chậm (4–6 MB/s). Dùng **US/EU**.
- **`export HF_HOME=$SCRIPT_DIR/.hf_cache`** — cache model trong workspace folder, persistent.
- **Không dùng `huggingface-cli download`** nếu không muốn login — để `transformers` tự download lần đầu khi chạy script.
- `touch ~/.no_auto_tmux` để tắt auto-tmux của Vast.ai (gây conflict với Termius).
- Model chỉ download 1 lần, sau đó load từ cache (~6 giây).

---

## Fine-tuning Method: LoRA (SFT)

- **Chỉ train 3.74% parameters** (`trainable: 119,734,272 / total: 3,205,672,960`)
- LoRA target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Config: `r=64, alpha=128, dropout=0.05`
- Full model frozen → chỉ LoRA adapter được cập nhật → tiết kiệm memory và tốc độ

## Memory

- VRAM peak: **~23.4 GB / 24 GB** (sát ngưỡng RTX 3090)
- Transformers tự cấp phát 90% memory cho model, 10% buffer tránh OOM
- `torch_dtype=bfloat16` → giảm memory xuống một nửa so với float32

---

## Training Speed (batch_size=8, non-CoT)

| Metric | Value |
|---|---|
| Tốc độ thực (tqdm) | **2.35s / mini-batch** |
| Tổng 25 batches (incl. overhead) | 80.6s |
| Overhead (data prep + save) | ~22s |
| Batch shape | `torch.Size([8, 240])` — sequence length thực tế ~240 tokens |
| Model load từ cache | ~6 giây |

> ⚠️ **Lưu ý quan trọng**: Số 3.22s/mini-batch trong output script bao gồm overhead (data prep, tokenization). Số thực của GPU là **2.35s/it** từ tqdm.

---

## Ước tính thời gian training (dựa trên 2.35s/step, batch_size=8)

| Job | Config | Steps | Ước tính |
|---|---|---|---|
| Pipeline pt1 Qwen | 3000s, 10ep, b=8 | 3750 | **2.4h** |
| Pipeline pt2 Qwen | 3000s, 10ep, b=2, CoT~3x | 45000 | **29.4h** |
| Pipeline pt1 Phi4 | 3000s, 10ep, b=8 | 3750 | **2.4h** |
| Pipeline pt2 Phi4 | 3000s, 10ep, b=2, CoT~3x | 45000 | **29.4h** |
| SFT Baseline ×2 | 3000s, 20ep, b=8 | 15000 | **9.8h** |
| Full Distill ×2 | 3000s, 20ep, b=2, CoT~3x | 90000 | **58.8h** |

> ⚠️ **Caveats quan trọng cho ước tính**:
> - pt2 và Full Distill dùng `batch_size=2` (không phải 8) → tốc độ/step sẽ **khác** (ít samples hơn/batch nhưng sequence CoT dài hơn nhiều)
> - CoT sequences dài hơn non-CoT ~3-5x → memory cao hơn, có thể chậm hơn/step
> - Cần timing test riêng cho CoT config để có số chính xác

---

## Accuracy & Loss (1 epoch, 200 samples)

- **Loss**: 2.921
- **Accuracy**: 33.12% sau 1 epoch
- Random baseline (10 lựa chọn): ~10%
- → Model đã học được signal sau chỉ 1 epoch với 200 samples, nhưng chưa có ý nghĩa thống kê

---

## Output Format

Model trả lời đúng format `[[number]]` ngay từ đầu (pretrained Qwen đã hiểu instruction):
```
assistant
[[9]]
```

---

## Lessons Learned (Vast.ai setup)

- **Tránh instance China** — HuggingFace download rất chậm (4–6 MB/s). Dùng **US/EU**.
- **`export HF_HOME=/workspace/.cache/huggingface`** trước khi chạy → model cache lưu vào `/workspace` (persistent volume), không mất khi restart instance.
- **Download model trước** bằng `huggingface-cli download Qwen/Qwen2.5-3B-Instruct` (interactive, thấy progress bar), sau đó mới `nohup` training.
- `tail -f` không hiện progress bar tqdm (dùng `\r`) — dùng `cat` để xem toàn bộ log, hoặc chạy trực tiếp (không nohup) để thấy realtime.
- `touch ~/.no_auto_tmux` để tắt auto-tmux của Vast.ai (gây conflict với Termius).
