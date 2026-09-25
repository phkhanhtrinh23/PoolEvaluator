#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mode="${MODE:-smoke}"
artifact_root="${POOLEVAL_ARTIFACT_ROOT:-artifacts}"
dataset_root="${POOLEVAL_DATASETS:-$repo_root/datasets}"
data_root="${POOLEVAL_DATA_ROOT:-$dataset_root/text2sql}"
bird_metadata_root="${BIRD_METADATA_ROOT:-$dataset_root/text2sql/bird}"
bird_database_root="${BIRD_DATABASE_ROOT:-$dataset_root/text2sql/bird}"

python -m pytest -q
python -m pooleval.models list --task all
python -m pooleval.models download --task all --cache-dir "${CHECKPOINT_DIR:-checkpoints}" --dry-run
python -m pooleval.models support --task all --cache-dir "${CHECKPOINT_DIR:-checkpoints}" --dry-run
python -m pooleval.pipeline smoke

# Prove that EX-Extended instances can be generated from each local benchmark.
python -m pooleval.databases \
  --dataset all \
  --split target \
  --limit-databases "${DB_SMOKE_LIMIT:-1}" \
  --output-dir "$artifact_root/db_smoke" \
  --text2sql-root "$data_root" \
  --bird-metadata-root "$bird_metadata_root" \
  --bird-database-root "$bird_database_root"

if [[ "$mode" != "full" ]]; then
  echo "Smoke run complete. Use MODE=full for real model inference."
  exit 0
fi

if [[ "${DOWNLOAD_MODELS:-0}" == "1" ]]; then
  python -m pooleval.models download --task all --cache-dir "${CHECKPOINT_DIR:-checkpoints}"
fi
if [[ "${DOWNLOAD_SUPPORT:-0}" == "1" ]]; then
  # Retrieval encoders (ColBERTv2, DINOv2) and local judges (Qwen2.5-VL-72B, GOFA).
  python -m pooleval.models support --task all --cache-dir "${CHECKPOINT_DIR:-checkpoints}"
fi
if [[ "${PREPARE_EXTERNAL_RUNTIMES:-0}" == "1" ]]; then
  python -m pooleval.models runtimes --task node --cache-dir "${CHECKPOINT_DIR:-checkpoints}"
fi

judge_args=()
if [[ "${RUN_JUDGES:-1}" == "1" ]]; then
  judge_args+=(--judge)
fi
common_args=(
  --dtype "${INFERENCE_DTYPE:-bfloat16}"
  --seed "${SEED:-42}"
  --checkpoint-dir "${CHECKPOINT_DIR:-checkpoints}"
  --artifact-root "$artifact_root"
)
if [[ -n "${SUBSET_BUDGET:-}" ]]; then
  common_args+=(--subset-budget "$SUBSET_BUDGET")  # 0 = random truncated-normal prior
fi
if [[ -n "${JUDGE_ROUNDS:-}" ]]; then
  common_args+=(--judge-rounds "$JUDGE_ROUNDS")
fi
if [[ "${REPEATS:-0}" != "0" ]]; then
  common_args+=(--repeats "$REPEATS")  # repeated runs over sampled pools with 95% CIs
fi
text2sql_judge_args=("${judge_args[@]}")
if [[ "${REQUIRE_ALL_JUDGES:-0}" == "1" ]]; then
  text2sql_judge_args+=(--require-all-judges)
fi

for dataset in ${TEXT2SQL_TARGETS:-spider bird}; do
  python -m pooleval.pipeline text2sql \
    --dataset "$dataset" \
    --target-items "${TARGET_ITEMS:-150}" \
    --source-candidates "${SOURCE_CANDIDATES:-2000}" \
    --max-models "${MAX_MODELS:-35}" \
    --text2sql-root "$data_root" \
    --bird-metadata-root "$bird_metadata_root" \
    --bird-database-root "$bird_database_root" \
    "${common_args[@]}" \
    "${text2sql_judge_args[@]}"
done

# Image and node pools are opt-in: they need their datasets under $dataset_root.
if [[ "${RUN_IMAGE:-0}" == "1" ]]; then
  for dataset in ${IMAGE_TARGETS:-usps svhn cifar10.1 cifar10-c imagenet-v2 imagenet-r imagenet-sketch}; do
    python -m pooleval.pipeline image \
      --dataset "$dataset" \
      --image-root "$dataset_root/image" \
      "${common_args[@]}" \
      "${judge_args[@]}"
  done
fi
if [[ "${RUN_NODE:-0}" == "1" ]]; then
  node_args=()
  if [[ "${SKIP_EXTERNAL:-0}" == "1" ]]; then
    node_args+=(--skip-external)
  fi
  for dataset in ${NODE_TARGETS:-acmv9 citationv1 dblpv7 ogbn-arxiv good-cora good-twitch good-webkb}; do
    python -m pooleval.pipeline node \
      --dataset "$dataset" \
      --node-root "$dataset_root/node" \
      "${common_args[@]}" \
      "${node_args[@]}" \
      "${judge_args[@]}"
  done
fi

# Loading all pools is opt-in because the paper's largest checkpoints require a
# multi-GPU host. Models are loaded and released sequentially.
if [[ "${LOAD_ALL_POOLS:-0}" == "1" ]]; then
  for task in text2sql image node; do
    python -m pooleval.models load \
      --task "$task" \
      --cache-dir "${CHECKPOINT_DIR:-checkpoints}" \
      --device-map auto \
      --dtype "${INFERENCE_DTYPE:-bfloat16}" \
      --seed "${SEED:-42}"
  done
fi

echo "Full PoolEvaluator run complete: $artifact_root"
