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
python -m pooleval.pipeline smoke

# Prove that EX-Extended instances can be generated from each local benchmark.
python -m pooleval.databases \
  --dataset all \
  --split target \
  --limit-databases "${DB_SMOKE_LIMIT:-1}" \
  --output-dir "$artifact_root/db_smoke" \
  --fusion-sql-root "$data_root" \
  --bird-metadata-root "$bird_metadata_root" \
  --bird-database-root "$bird_database_root"

if [[ "$mode" != "full" ]]; then
  echo "Smoke run complete. Use MODE=full for real model inference."
  exit 0
fi

if [[ "${DOWNLOAD_MODELS:-0}" == "1" ]]; then
  python -m pooleval.models download --task all --cache-dir "${CHECKPOINT_DIR:-checkpoints}"
fi
if [[ "${PREPARE_EXTERNAL_RUNTIMES:-0}" == "1" ]]; then
  python -m pooleval.models runtimes --task node --cache-dir "${CHECKPOINT_DIR:-checkpoints}"
fi

judge_args=()
if [[ "${RUN_JUDGES:-1}" == "1" ]]; then
  judge_args+=(--judge)
fi
if [[ "${REQUIRE_ALL_JUDGES:-0}" == "1" ]]; then
  judge_args+=(--require-all-judges)
fi

for dataset in spider bird; do
  python -m pooleval.pipeline text2sql \
    --dataset "$dataset" \
    --target-items "${TARGET_ITEMS:-150}" \
    --source-candidates "${SOURCE_CANDIDATES:-2000}" \
    --max-models "${MAX_MODELS:-35}" \
    --checkpoint-dir "${CHECKPOINT_DIR:-checkpoints}" \
    --artifact-root "$artifact_root" \
    --fusion-sql-root "$data_root" \
    --bird-metadata-root "$bird_metadata_root" \
    --bird-database-root "$bird_database_root" \
    "${judge_args[@]}"
done

# Loading all pools is opt-in because the paper's largest checkpoints require a
# multi-GPU host. Models are loaded and released sequentially.
if [[ "${LOAD_ALL_POOLS:-0}" == "1" ]]; then
  for task in text2sql image node; do
    python -m pooleval.models load \
      --task "$task" \
      --cache-dir "${CHECKPOINT_DIR:-checkpoints}" \
      --device-map auto
  done
fi

echo "Full PoolEvaluator run complete: $artifact_root"
