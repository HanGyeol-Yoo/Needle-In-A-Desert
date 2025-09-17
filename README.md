# Needle-In-A-Desert

## How To Use
```
uv venv needle_venv
source needle_venv/bin/activate
uv pip install numpy matplotlib seaborn datasets transformers torch
```
Modify the `configs.yaml` as follows and then run `bash pipeline.sh`.
```yaml
# 사용할 모델의 전체 경로
model_path: "Huggingface/Model_Repo"

# Chat template 사용 여부 (true/false)
use_chat_template: true

# 모든 결과물이 저장될 최상위 디렉토리
output_dir: "./Model_Repo"

# 데이터셋 생성을 위한 원본 파일 경로
needles_path: "./needles.jsonl"
en_jsonl: "./EN.jsonl"
kr_jsonl: "./KR.jsonl"

# --- 데이터셋 생성 옵션 ---
len_start: 1000
len_end: 33000
len_step: 2000
repeats_per_combo: 10

# --- 추론 옵션 ---
# 사용할 GPU 개수
num_gpus: 4
# 생성할 최대 토큰 수
max_new_tokens: 128
```
