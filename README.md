# UIT_IT2024.CH201 — Complexity-Aware Fine-Tuning

## Giới thiệu

- **Tên môn học:** Large Language Models / Chủ đề thực nghiệm với mô hình ngôn ngữ lớn
- **Mã môn:** UIT_IT2024.CH201
- **Bài báo:** Complexity-Aware Fine-Tuning

### Học viên

| STT | Họ tên |  
|:---:|--------|
| 1 | Trần Trọng Đạt |  
| 2 | Trần Minh Hiếu |
| 3 | Lê Thành Duy Ân | 
| 4 | Nguyễn Văn Phú | 

## Tên đề tài

**Reproduce và phân tích phương pháp Complexity-Aware Fine-Tuning cho mô hình ngôn ngữ lớn**

## Tổng quan dự án

Dự án tập trung tìm hiểu và tái hiện một phần thực nghiệm của bài báo **Complexity-Aware Fine-Tuning**. Bài báo đề xuất một hướng fine-tuning hiệu quả hơn cho các mô hình ngôn ngữ nhỏ bằng cách không sử dụng Chain-of-Thought (CoT) cho toàn bộ dữ liệu huấn luyện.

Thay vào đó, phương pháp sử dụng **entropy** để ước lượng độ khó của từng câu hỏi. Những câu dễ và trung bình được huấn luyện bằng Supervised Fine-Tuning (SFT) với đáp án ngắn, trong khi những câu khó mới được huấn luyện với lời giải thích từng bước từ mô hình giáo viên.

Mục tiêu chính của dự án là hiểu cách mã nguồn GitHub hiện thực hóa pipeline của bài báo, chạy lại một số kịch bản thực nghiệm quan trọng và so sánh kết quả giữa phương pháp đề xuất với các baseline.

## Mục tiêu chính

- Tìm hiểu ý tưởng chính của Complexity-Aware Fine-Tuning (CAFT)
- Phân tích cách bài báo dùng entropy để chia dữ liệu thành Easy, Middle và Hard
- Chạy lại pipeline chính trên MMLU-Pro với Qwen2.5-3B
- So sánh CAFT với các baseline như SFT, Full Distillation và Curriculum nếu có đủ tài nguyên GPU
- Ghi nhận các lưu ý khi chạy thực nghiệm, đặc biệt là `max_new_tokens`, epoch, GPU và chi phí tính toán

## Hướng tiếp cận dự kiến

Dự án sử dụng mã nguồn chính thức của tác giả tại repository:

```text
https://github.com/LabARSS/complexity-aware-fine-tuning
```

Pipeline chính của bài báo gồm hai giai đoạn:

```text
Stage 1:
Easy + Middle questions -> SFT với đáp án ngắn

Stage 2:
Hard questions -> SFT với Chain-of-Thought từ teacher model
```

Với MMLU-Pro và Qwen2.5-3B, hai script chính là:

```bash
PYTHONPATH=src python src/experiments/pipeline/pipeline/pipeline_pt1_qwen3b.py
PYTHONPATH=src python src/experiments/pipeline/pipeline/pipeline_pt2_qwen3b.py
```

Nhóm đã ưu tiên chạy trên MMLU-Pro trước vì đây là dataset được repo hỗ trợ đầy đủ nhất cho Qwen2.5-3B, bao gồm pipeline chính và các baseline.

## Cấu trúc mã nguồn liên quan

Các thư mục quan trọng trong repository:

- `data/source/`: dữ liệu gốc
- `data/data_splits/`: dữ liệu đã chia thành Easy, Middle, Hard, Combined, Validation và Test
- `src/experiments/single_token_entropy/`: script tính entropy để ước lượng độ khó
- `src/experiments/pipeline/`: thực nghiệm trên MMLU-Pro
- `src/experiments/pipeline_gsm8k/`: thực nghiệm trên GSM8K
- `src/experiments/pipeline_medmcqa/`: thực nghiệm trên MedMCQA
- `src/reasoning_fine_tune/training/sft/`: logic huấn luyện dùng chung

## Các kịch bản thực nghiệm

### 1. CAFT / Ours Pipeline

Đây là phương pháp chính của bài báo.

```text
Easy + Middle -> direct SFT
Hard -> CoT distillation
```

Script Qwen2.5-3B trên MMLU-Pro:

```bash
PYTHONPATH=src python src/experiments/pipeline/pipeline/pipeline_pt1_qwen3b.py
PYTHONPATH=src python src/experiments/pipeline/pipeline/pipeline_pt2_qwen3b.py
```

### 2. SFT Baseline

Baseline này huấn luyện toàn bộ dữ liệu với đáp án ngắn, không chia stage và không dùng CoT.

```bash
PYTHONPATH=src python src/experiments/pipeline/sft_baseline/sft_baseline_qwen3b.py
```

### 3. Full Distillation Baseline

Baseline này dùng CoT cho toàn bộ dữ liệu. Đây là baseline mạnh nhưng tốn nhiều token và chi phí huấn luyện.

```bash
PYTHONPATH=src python src/experiments/pipeline/full_distill/full_distill_baseline_qwen3b.py
```

### 4. Curriculum Baseline

Curriculum train dữ liệu theo thứ tự Easy -> Middle -> Hard, nhưng vẫn dùng SFT với đáp án ngắn. Baseline này giúp kiểm tra liệu CAFT có tốt hơn việc chỉ sắp xếp dữ liệu theo độ khó hay không.

```bash
PYTHONPATH=src python src/experiments/pipeline/sft_curriculum_baseline/sft_curriculum_baseline_qwen.py
```

## Bộ dữ liệu sử dụng

Bài báo sử dụng ba dataset chính:

- `MMLU-Pro`: tập câu hỏi trắc nghiệm nhiều lĩnh vực
- `GSM8K`: tập bài toán suy luận toán học
- `MedMCQA`: tập câu hỏi trắc nghiệm y khoa

Trong phạm vi reproduce ban đầu, nhóm ưu tiên:

```text
Dataset: MMLU-Pro
Model: Qwen2.5-3B-Instruct
Epochs: 5
```

Lý do là MMLU-Pro có đầy đủ script cho Qwen2.5-3B trong repo, bao gồm pipeline chính và các baseline quan trọng.

## Lưu ý khi chạy thực nghiệm

### `max_new_tokens`

Tham số `max_new_tokens` quyết định số token tối đa model được sinh ra khi evaluate.

Trong bài báo:

```text
MMLU-Pro:
SFT / Pipeline easy-middle: 30
Pipeline hard / Distillation: 8192
```

Với SFT hoặc Easy/Middle, output chỉ là đáp án ngắn như `[[3]]`, nên 30 token là đủ.

Với Hard + CoT hoặc Full Distillation, output có thể là lời giải dài, nên paper dùng 8192 token. Nếu giảm xuống 2048 để tiết kiệm chi phí, một số CoT dài có thể bị cắt, làm giảm accuracy.

### GPU

Nên chạy trên GPU CUDA đủ mạnh:

- A10 / A10G 24GB: có thể chạy
- L40S 48GB: tốt
- A100 40GB/80GB: phù hợp nhất

Không nên dùng P100 nếu bản PyTorch hiện tại không hỗ trợ compute capability `sm_60`.

### Lưu log

Khi chạy trên server thuê GPU, nên lưu log:

```bash
mkdir -p logs
PYTHONPATH=src python script.py 2>&1 | tee logs/run.log
```

Kết quả accuracy thường nằm trong:

```bash
find artifacts -name "*results.jsonl"
```

## Kết quả kỳ vọng

Dự án hướng đến các kết quả sau:

- Hiểu rõ pipeline CAFT và cách repo hiện thực hóa bài báo
- Có kết quả chạy lại trên MMLU-Pro với Qwen2.5-3B
- So sánh được CAFT với SFT baseline và Full Distillation nếu đủ GPU
- Phân tích được trade-off giữa accuracy và chi phí token
- Chuẩn bị được nội dung demo và slide báo cáo cho lớp UIT_IT2024.CH201

## Ghi chú

`README.md` này dùng để tóm tắt hướng reproduce và trình bày thực nghiệm của nhóm. Kết quả accuracy cụ thể có thể thay đổi tùy GPU, số epoch, `max_new_tokens`, batch size và checkpoint được chọn.

Nếu chỉ demo nhanh, có thể dùng mini version với model nhỏ hơn như `Qwen/Qwen2.5-0.5B-Instruct`, ít sample và 1 epoch. Mini version chỉ dùng để minh họa pipeline, không dùng để so sánh trực tiếp với kết quả trong paper.
