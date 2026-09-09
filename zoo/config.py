"""Zoo configuration: data paths, the model manifest (provenance groups), and
inference/eval knobs. See zoo/README.md for the design rationale.

The manifest gives each pool member a real base checkpoint (`model`) and a
provenance `group` (the base family). Members that share a group but differ only in
`prompt`/`temperature` are the paper's within-group NEAR-CLONES (correlated errors);
members in different groups are the independent evidence the joint estimator exploits.
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

# --- data paths (FusionSQL layout on this machine) ---
DATA_ROOT = "/mnt/win_d/data_FusionSQL"
SPIDER_DEV = os.path.join(DATA_ROOT, "spider", "sft_spider_dev_text2sql.json")
SPIDER_TRAIN = os.path.join(DATA_ROOT, "spider", "sft_spider_train_text2sql.json")
SPIDER_DB_DIRS = [os.path.join(DATA_ROOT, "spider", "database"),
                  os.path.join(DATA_ROOT, "spider", "test_database")]

# where generations / results / poolruns are cached (in the code repo, gitignored)
ARTIFACT_ROOT = os.environ.get(
    "POOLEVAL_ARTIFACT_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "zoo_artifacts"),
)


@dataclass
class ZooMember:
    name: str            # unique pool-member id
    model: str           # API model id
    group: str           # provenance group (base family) -> ground-truth G(m)
    provider: str = "openai"
    prompt: str = "schema"   # prompt style: schema | minimal | fewshot
    temperature: float = 0.0


# M=10 members across 7 provenance groups. Three groups carry a near-clone pair
# (same base, different prompt/decoding) -> true independent count ~= 7 < M=10,
# exactly the correlated-pool regime PoolEval-SQL targets.
DEFAULT_MANIFEST: List[ZooMember] = [
    # group gpt-4o (near-clone pair: schema-rich vs minimal prompt)
    ZooMember("gpt4o-schema",     "gpt-4o",          "gpt-4o",       prompt="schema"),
    ZooMember("gpt4o-minimal",    "gpt-4o",          "gpt-4o",       prompt="minimal"),
    # group gpt-4o-mini (near-clone pair: schema vs few-shot)
    ZooMember("gpt4omini-schema", "gpt-4o-mini",     "gpt-4o-mini",  prompt="schema"),
    ZooMember("gpt4omini-fewshot","gpt-4o-mini",     "gpt-4o-mini",  prompt="fewshot"),
    # singleton groups (independent bases)
    ZooMember("gpt41",            "gpt-4.1",         "gpt-4.1",      prompt="schema"),
    ZooMember("gpt41-mini",       "gpt-4.1-mini",    "gpt-4.1-mini", prompt="schema"),
    ZooMember("gpt41-nano",       "gpt-4.1-nano",    "gpt-4.1-nano", prompt="schema"),
    ZooMember("gpt4-turbo",       "gpt-4-turbo",     "gpt-4-turbo",  prompt="schema"),
    # group gpt-3.5 (weak base, near-clone pair -> a colluding low-accuracy clique)
    ZooMember("gpt35-schema",     "gpt-3.5-turbo",   "gpt-3.5-turbo",prompt="schema"),
    ZooMember("gpt35-fewshot",    "gpt-3.5-turbo",   "gpt-3.5-turbo",prompt="fewshot"),
]


@dataclass
class ZooConfig:
    manifest: List[ZooMember] = field(default_factory=lambda: list(DEFAULT_MANIFEST))
    n_target: int = 150       # target (dev) items to evaluate  [user choice: ~150]
    n_source: int = 120       # source (train) items for the seen-prior calibration
    seed: int = 0
    exec_timeout: float = 5.0  # per-query wall-clock seconds
    max_tokens: int = 512

    @property
    def groups(self) -> List[str]:
        seen, out = set(), []
        for m in self.manifest:
            if m.group not in seen:
                seen.add(m.group); out.append(m.group)
        return out

    def group_ids(self) -> List[int]:
        gidx = {g: i for i, g in enumerate(self.groups)}
        return [gidx[m.group] for m in self.manifest]
