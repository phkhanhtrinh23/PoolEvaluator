"""SynSQL-2.5M as a RETRIEVABLE PRIOR CORPUS.

Motivation (see experiments/SYNSQL_PRIOR.md): PoolEval-SQL's "seen prior" is the
only anchor that fixes the gauge, yet in every run so far it came from an arbitrary
*labeled split of the target benchmark itself* -- exactly the thing a deployed pool
operator does not have. This package replaces it with a prior estimated on
SynSQL-2.5M subsets RETRIEVED to match the unlabeled target, so the anchor needs no
target labels at all.

Layout on this machine:
  data.json        9.3 GB, 2.5M records {db_id, question, sql, sql_complexity,
                   question_style, external_knowledge, cot}
  databases.zip -> 16,583 SQLite databases extracted to SYNSQL_DB_ROOT
"""
import os

_SNAP = ("/mnt/win_d/hub/datasets--seeklhy--SynSQL-2.5M/snapshots/"
         "cca2c84cc3b41afa6b51534762a6a3a4a420baca")
SYNSQL_DATA = os.path.join(_SNAP, "data.json")
SYNSQL_TABLES = os.path.join(_SNAP, "tables.json")
SYNSQL_DB_ROOT = "/mnt/win_d/synsql_databases"

# big derived artifacts live off-repo (the repo is in Dropbox)
CACHE_ROOT = "/mnt/win_d/synsql_cache"
INDEX_JSONL = os.path.join(CACHE_ROOT, "index.jsonl")      # minimal fields, 1 line/record
SUBSETS_DIR = os.path.join(CACHE_ROOT, "subsets")          # partitionings

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "results")
ARTIFACTS = os.path.join(REPO_ROOT, "zoo_artifacts")       # gitignored

# ---- experiment knobs ----
SUBSET_SIZE = 1000        # "around 1K samples per subset" (user spec)
N_CANDIDATES = 32         # candidate subsets we pay to probe with the model pool
N_PROBE = 25              # labeled probe items per candidate subset
TOP_K = 5                 # retrieved subsets -> prior (5 x 25 = 125 items, matching
                          # the n_source=120 budget the in-split prior already uses)
SEED = 0
