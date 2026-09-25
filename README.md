# PoolEvaluator

This is the paper-aligned artifact for **"Which Model Should Be Chosen? Joint Performance Estimation and Ranking of Black-Box Models on Unlabeled Data"**. It jointly estimates and ranks models from their agreement on an unlabeled target set, initializes the estimate from retrieved labeled subsets, and optionally asks a small ensemble of external judges to resolve ambiguous items.

![PoolEvaluator pipeline](assets/pipeline.png)

<!-- ![Motivation: estimation and ranking error versus latency](assets/motivation.png) -->

## What is included

- The exact appendix pool sizes: **35 Text2SQL**, **20 image-classification**, and **20 node-classification** members in [`model_pools/`](model_pools/).
- Lazy download/loading for Transformers, timm, PyTorch Geometric, zero-shot vision, and graph adapter/projector repositories in [`pooleval/models.py`](pooleval/models.py).
- Stage 1 top-K subset retrieval, Stage 2 closed-form EM over (α<sub>j</sub>, β<sub>j</sub>, γ<sub>j</sub>), and Stage 3 warm-started judge validation.
- EX-Extended evaluation on the original database plus **five modified instances** for SQLite, PostgreSQL, and MySQL, with deterministic caching and integrity checks.
- Stage 1 encoders: ColBERTv2 for Text2SQL questions, DINOv2 for images, and parameter-free feature propagation for graph nodes.
- Text2SQL judges GPT-5.4, GPT-5.5, and Claude Opus 4.5 with majority voting, ties broken by accuracy-weighted support. Local judges: Qwen2.5-VL-72B-Instruct for images and GOFA for graphs.
- End-to-end pipelines for Text2SQL (Spider, BIRD, Spider 2.0, BEAVER, ScienceBenchmark, EntSQL, LiveSQLBench), image classification (seven targets), and node classification (seven targets).
- Repeated runs over sampled 5–30-model pools with 95% confidence intervals, per-round EM iteration and error logs, and end-to-end latency.
- Node pools with three training seeds per architecture and runners for GraphGPT, GraphPFN, and LLaGA in their official runtimes.
- One shell entry point for tests, dry-run model preparation, database generation, and full evaluation.

The two manuscript figures are committed as README assets. PDFs and exploratory reports are deliberately absent from `main`. The paper itself remains the source of truth for tables and derivations.

## 1. Create the environment

Python 3.10 or newer is required.

```bash
git clone <repository-url>
cd pool_text2sql_eval_code

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Core estimator, model-pool definitions, database tooling, and tests
python -m pip install -e '.[dev]'

# Add model runtimes and API clients on an experiment machine
python -m pip install -e '.[models,judges]'

# Optional: ogbn-arxiv needs OGB; GOOD targets need the GOOD package
python -m pip install -e '.[graph-data]'
python -m pip install git+https://github.com/divelab/GOOD.git

# Optional: PostgreSQL (ScienceBenchmark) and MySQL (BEAVER) clients
python -m pip install -e '.[sql-servers]'
```

`torch-geometric` may require a CUDA-specific wheel on older systems. Follow the [PyG installation matrix](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if the normal extra does not match the installed PyTorch/CUDA build.

## 2. Configure the datasets

Keep downloaded data outside Git under one root. The paths below are the convention used throughout this README:

```bash
export POOLEVAL_DATASETS="$PWD/datasets"
mkdir -p "$POOLEVAL_DATASETS"/{text2sql,image,node}
```

### Text2SQL datasets

| Dataset | Official source | Save or extract to | Database | Setup |
|---|---|---|---|---|
| Spider | [project and download page](https://yale-lily.github.io/spider), [evaluation repository](https://github.com/taoyds/spider) | `datasets/text2sql/spider/` | SQLite | Download the data archive from the project page. |
| BIRD | [project and download page](https://bird-bench.github.io/) | `datasets/text2sql/bird/` | SQLite | The project page links the train/dev database packages. |
| Spider 2.0 | [official repository](https://github.com/xlang-ai/Spider2), [project page](https://spider2-sql.github.io/) | `datasets/text2sql/spider2/` | SQLite | Clone the repository (its `spider2-lite/` folder) and unzip the local databases into `spider2-lite/resource/databases/spider2-localdb/`. |
| BEAVER | [official repository](https://github.com/beaverbench/beaver), [dataset](https://huggingface.co/collections/beaverbench/beaver-dataset) | `datasets/text2sql/beaver/<dw,dw_real,nova,neutron>/dev.json` | MySQL | Run the repository's `data/download_hf.py` (authenticated) and load `beaver_db.zip` into MySQL as described on the `beaver-table` dataset page. |
| ScienceBenchmark | [official dataset page](https://sciencebenchmark.cloudlab.zhaw.ch/), [dataset repository](https://github.com/ckosten/sciencebenchmark_dataset) | `datasets/text2sql/sciencebenchmark/<cordis,oncomx,sdss>/` | PostgreSQL | Copy each domain folder (`dev.json`, `seed.json`, `synth.json`, `tables.json`) and `pg_restore` each dump into a database named after the domain (with `CREATE EXTENSION pg_trgm`). |
| EntSQL | [paper](https://arxiv.org/abs/2606.03363), [dataset](https://huggingface.co/datasets/XuWave/Rethinking_Enterprise_Text_to_SQL) | `datasets/text2sql/entsql/` | SQLite | Download `JSON/` and unzip every `DB/<Domain>.zip` into `DB/<Domain>/`. Gold goes in `gold/<id>.sql` or `gold/<id>.csv`. |
| LiveSQLBench | [official repository](https://github.com/bird-bench/livesqlbench), [SQLite dataset](https://huggingface.co/datasets/birdsql/livesqlbench-base-lite-sqlite) | `datasets/text2sql/livesqlbench/` | SQLite | Clone the SQLite release and merge its ground-truth file into `livesqlbench_data_sqlite.jsonl` with the repository's `integrate_gt_data.py`. |

Spider and BIRD use the following layout, with each metadata file a JSON list whose rows contain `db_id`, `question`, and `sql` (`schema` and `evidence` are optional):

```text
datasets/text2sql/
├── spider/
│   ├── sft_spider_dev_text2sql.json
│   ├── sft_spider_train_text2sql.json
│   ├── database/<db_id>/<db_id>.sqlite
│   └── test_database/<db_id>/<db_id>.sqlite
└── bird/
    ├── sft_bird_dev_text2sql.json
    ├── sft_bird_train_text2sql.json
    ├── dev/dev_databases/<db_id>/<db_id>.sqlite
    └── train/train_databases/<db_id>/<db_id>.sqlite
```

The other five benchmarks are read in their official release layouts by [`pooleval/benchmarks.py`](pooleval/benchmarks.py):

- **Spider 2.0.** Its local SQLite instances are scored with EX-Extended against the gold SQL. Instances released with gold result tables are scored with the official Spider 2.0 table matcher (`condition_cols`, `ignore_order`, alternative gold tables).
- **EntSQL.** Each question is paired with its long document as evidence. Result-table gold is scored with the official EntSQL row matcher.
- **LiveSQLBench.** Its SELECT tasks use the knowledge-base entries they reference as evidence.
- **BEAVER and ScienceBenchmark.** Their schemas come from the database and from `tables.json`, respectively.

Benchmarks without a labeled training split take their Stage 1 calibration subsets from the Spider and BIRD training sets (`tasks.text2sql.calibration_sources`). ScienceBenchmark calibrates on its own seed and synthetic splits.

PostgreSQL and MySQL databases are addressed as `postgresql://<database>` and `mysql://<database>`. Connection settings come from the environment:

```bash
export PGHOST=localhost PGPORT=5432 PGUSER=postgres PGPASSWORD=...        # ScienceBenchmark
export POOLEVAL_MYSQL_HOST=localhost POOLEVAL_MYSQL_PORT=3306 \
       POOLEVAL_MYSQL_USER=root POOLEVAL_MYSQL_PASSWORD=...                # BEAVER
```

`tasks.text2sql.database_names` maps a benchmark database id to a server database with a different name. Point the pipeline at the dataset root with:

```bash
export POOLEVAL_DATA_ROOT="$POOLEVAL_DATASETS/text2sql"
export BIRD_METADATA_ROOT="$POOLEVAL_DATASETS/text2sql/bird"
export BIRD_DATABASE_ROOT="$POOLEVAL_DATASETS/text2sql/bird"
export POOLEVAL_ARTIFACT_ROOT="$PWD/artifacts"
```

The portable defaults use `datasets/text2sql/` inside the repository. For a dataset stored elsewhere, pass `--text2sql-root`, `--bird-metadata-root`, and `--bird-database-root` to the Python entry points, or set the environment variables above when using `scripts/run_all.sh`. The SQLite loaders skip Git-LFS pointer stubs and require materialized database files larger than 4 KiB.

### Image-classification datasets

The paper uses MNIST, CIFAR-10, and ImageNet as labeled source datasets and evaluates on the seven target datasets shown below.

| Role | Dataset | Official source | Save or extract to |
|---|---|---|---|
| Source | MNIST | [official dataset page](https://yann.lecun.org/exdb/mnist/) | `datasets/image/mnist/` |
| Source | CIFAR-10 | [official dataset page](https://www.cs.toronto.edu/~kriz/cifar.html) | `datasets/image/cifar10/` |
| Source | ImageNet-1K (ILSVRC 2012) | [official download page](https://www.image-net.org/download.php) | `datasets/image/imagenet/` with `train/` and `val/` class directories |
| Target | USPS | [torchvision source and download URLs](https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/usps.html) | `datasets/image/usps/` |
| Target | SVHN | [official dataset page](http://ufldl.stanford.edu/housenumbers/) | `datasets/image/svhn/` |
| Target | CIFAR-10.1 | [official repository](https://github.com/modestyachts/CIFAR-10.1) | `datasets/image/cifar10.1/` |
| Target | CIFAR-10-C | [official Zenodo record](https://zenodo.org/records/2535967), [repository](https://github.com/hendrycks/robustness) | `datasets/image/cifar10-c/` |
| Target | ImageNet-V2 | [official repository](https://github.com/modestyachts/ImageNetV2) | `datasets/image/imagenet-v2/` |
| Target | ImageNet-R | [official repository](https://github.com/hendrycks/imagenet-r) | `datasets/image/imagenet-r/` |
| Target | ImageNet-Sketch | [official repository](https://github.com/HaohanWang/ImageNet-Sketch) | `datasets/image/imagenet-sketch/` |

MNIST, CIFAR-10, USPS, SVHN, and CIFAR-10.1 are downloaded automatically into their listed directories on first use. Extract CIFAR-10-C so that `datasets/image/cifar10-c/` (or its `CIFAR-10-C/` subfolder) holds `labels.npy` and one `<corruption>.npy` per corruption. ImageNet requires accepting its access terms and must be arranged as `imagenet/val/<wnid>/*.JPEG`. The loader takes the official ImageNet-1K index order from the 1000 sorted WordNet ids of that folder and maps ImageNet-R and ImageNet-Sketch folders (`<wnid>/`) and ImageNet-V2 folders (`0`–`999`) onto it. It never re-sorts a target's folders independently. ImageNet-R predictions are restricted to its 200 classes. Single wrapper directories left by archives, such as `imagenet-r/imagenet-r/`, are skipped automatically.

### Node-classification datasets

| Dataset | Official source | Save or extract to | Access notes |
|---|---|---|---|
| ACMv9, Citationv1, DBLPv7 | [GNNEvaluator repository and dataset link](https://github.com/Amanda-Zheng/GNNEvaluator) | `datasets/node/gnnevaluator/{acmv9,citationv1,dblpv7}/` | Use the Google Drive folder linked under **Instructions** in the repository. The loader finds `<name>_docs.txt`, `<name>_edgelist.txt`, and `<name>_labels.txt` anywhere below the listed folder. |
| ogbn-arxiv | [official OGB documentation](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) | `datasets/node/ogb/` | `PygNodePropPredDataset(name="ogbn-arxiv", root=...)` downloads it automatically. |
| GOOD-Cora, GOOD-Twitch, GOOD-WebKB | [official GOOD repository](https://github.com/divelab/GOOD), [dataset API](https://good.readthedocs.io/en/latest/_autosummary/GOOD.data.good_datasets.html) | `datasets/node/good/{GOODCora,GOODTwitch,GOODWebKB}/` | Use the predefined GOOD domain and shift splits from the paper experiment. |

The node pipeline trains the pool on labeled source nodes and evaluates on the target nodes. For ACMv9, Citationv1, and DBLPv7, the pool trains on a different source graph set in `tasks.node.sources` (default: ACMv9 for the other two, DBLPv7 for ACMv9) and uses GNNEvaluator's 70/30 source split. For ogbn-arxiv it trains on the OGB train split and evaluates on the test split. For GOOD it trains on the train split and evaluates on the OOD test split of the domain and shift set in `tasks.node.good`.

GOFA and the graph language models read node text, and GOFA prompts name the candidate labels. A dataset folder supplies them in `node_text.json` (a list aligned with node indices) and `class_names.json`. ogbn-arxiv gets its category names and title/abstract text automatically. For the GOOD targets, build both files from the public raw releases (downloads are cached under `datasets/node/raw_text/`):

```bash
python -m pooleval.node_metadata --dataset all --node-root "$POOLEVAL_DATASETS/node"
```

| Target | Node text | Class names |
|---|---|---|
| GOOD-Cora | Title and abstract from the original Cora (`cora-classify`) release, matched by paper id | The 70 Cora topic paths |
| GOOD-WebKB | Page title and text from the CMU WebKB archive, matched through each node's LINQS word vector and URL | Per university, with each node's university |
| GOOD-Twitch | A description from SNAP account metadata (language, account age, views, partner status, number of games) | "does not stream mature content", "streams mature content" |

Each build also writes `node_meta.json` with the node count and a SHA-1 of the label sequence, and the loaders use the files only for a graph with the same labels.

## 3. Inspect, download, and load the paper model pools

First inspect the complete model pools without downloading weights:

```bash
python -m pooleval.models list --task all
python -m pooleval.models download --task all --dry-run
```

Download one pool or all three. Gated repositories require a Hugging Face token and license acceptance.

```bash
export HF_TOKEN=hf_...
python -m pooleval.models download --task text2sql --cache-dir checkpoints
python -m pooleval.models download --task image    --cache-dir checkpoints
python -m pooleval.models download --task node     --cache-dir checkpoints
# equivalent: --task all

# Clone the official GraphGPT, GraphPFN, and LLaGA runtime source trees.
python -m pooleval.models runtimes --task node --cache-dir checkpoints
```

Download the retrieval encoders and the local judges (`--dry-run` lists them first). Text2SQL uses ColBERTv2. Images use DINOv2-small and Qwen2.5-VL-72B-Instruct. Graphs use GOFA: its source tree, the `mistral_qamag03_best_ckpt.pth` checkpoint, the ICAE weights, and the gated `mistralai/Mistral-7B-Instruct-v0.2` backbone (accept its license and set `HF_TOKEN`). `--load` loads each one once to verify it.

```bash
python -m pooleval.models support --task all --cache-dir checkpoints --dry-run
python -m pooleval.models support --task all --cache-dir checkpoints
python -m pooleval.models support --task text2sql --cache-dir checkpoints --load
```

Verify runtime loading sequentially (a loaded member is released before the next):

```bash
python -m pooleval.models load --task text2sql --cache-dir checkpoints
python -m pooleval.models load --task image    --cache-dir checkpoints
python -m pooleval.models load --task node     --cache-dir checkpoints
```

The large 70B/72B checkpoints need a multi-GPU machine. `device_map=auto` is used. Local Transformers, timm, and PyG inference explicitly requests bfloat16 on supported CUDA hardware and safely falls back to float32 elsewhere. Pass `--dtype float32` to force float32 or `--dtype auto` to delegate dtype selection to Transformers. `ModelPool.iter_load()` is the programmatic entry point. [`pooleval/adapters.py`](pooleval/adapters.py) turns loaded members into Text2SQL, image, or node predictions. The two LLaGA entries consist of a Vicuna base plus a released projector. GraphGPT additionally downloads its released graph encoder. GraphGPT, GraphPFN, and LLaGA retain their official runtimes because their custom graph types are not supported by Transformers `AutoModel`. The node pipeline runs them through the external-runner protocol in section 6. GraphPFN's public repository contains graph-adapter weights, but its required LimiX backbone has a separate license. Obtain that backbone under its upstream terms before running GraphPFN. Local checkpoint paths are retained in `LoadedModel.local_path`.

## 4. Create EX-Extended database instances

Build five modified instances for **every unique database** referenced by both target sets:

```bash
python -m pooleval.databases \
  --dataset all \
  --split target \
  --output-dir artifacts/db_instances
```

For a cheap validation first:

```bash
python -m pooleval.databases --dataset spider --limit-databases 1 --output-dir artifacts/db_smoke
python -m pooleval.databases --dataset bird   --limit-databases 1 --output-dir artifacts/db_smoke
```

Each SQLite database gets `instance_1.sqlite` through `instance_5.sqlite` plus an `instances.json` metadata file. The variants make deterministic value replacements, deletions, and insertions, then run `PRAGMA integrity_check`. PostgreSQL and MySQL databases get the same five variants as server databases named `<database>__pe1` to `<database>__pe5`: PostgreSQL copies are `CREATE DATABASE ... TEMPLATE` clones and MySQL copies are table-by-table. Finished instances are marked and reused on later runs. Source databases are opened read-only by the evaluator and are never modified.

## 5. Configure the judges

**Text2SQL.** Set the provider credentials before enabling Stage 3:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

GPT-5.4 and GPT-5.5 use the OpenAI Responses API with strict JSON-schema output. Claude Opus 4.5 (`claude-opus-4-5`) uses Anthropic's Messages API with JSON-schema output. The members are listed under `judges.text2sql.members` in `configs/paper.yaml`. The ensemble takes a majority vote over candidate answers and NONE. A tie goes to the tied choice with the largest accuracy-weighted support: the summed current α of the pool models that produced it. NONE is backed by no model, so its support is zero. Any remaining tie keeps the configured judge order. By default a missing provider is reported and the available judges continue. Pass `--require-all-judges` for strict three-member behavior.

**Image.** Qwen2.5-VL-72B-Instruct is loaded from `checkpoints/judges/` with Transformers and `device_map=auto`. In bfloat16 its weights need about 145 GB of GPU memory. With less, Accelerate offloads the remainder to CPU, which is slower.

**Node.** GOFA pins its own dependencies, so it runs in its own environment. Create that environment from `checkpoints/runtimes/GOFA/environment.yml` and set `judges.node.python` to its interpreter. `pooleval.local_judges.GOFAJudge` then starts [`scripts/runners/gofa_judge.py`](scripts/runners/gofa_judge.py) once per run and sends one query per judge round. Each query is the target node, up to `max_neighbors` neighbors, and a question node listing the candidate labels plus "None of the above". GOFA needs node text (section 2).

Every judge picks one of the pool's valid candidate answers or NONE. An output that can never be correct is not a candidate: a SQL query that fails on the original database or any of the five modified instances, or a node answer a model could not map to a class. An item with no valid candidate is never sent to a judge.

## 6. Run PoolEvaluator

Run the CPU-only estimator smoke test:

```bash
python -m pooleval.pipeline smoke
```

Run a small real Spider experiment using two target items and a subset of models:

```bash
python -m pooleval.pipeline text2sql \
  --dataset spider \
  --target-items 2 \
  --source-candidates 100 \
  --max-models 3 \
  --checkpoint-dir checkpoints
```

Enable judge refinement by adding `--judge`. Predictions, modified databases, judge votes, and reports are cached under `artifacts/`, so interrupted runs are resumable. The Text2SQL `--dataset` choices are `spider`, `bird`, `spider2`, `beaver`, `sciencebenchmark`, `entsql` (with `--language en|zh`), and `livesqlbench`.

`--subset-budget K` overrides the configured K for any task. With `--subset-budget 0`, retrieval and calibration are skipped. Every coordinate of θ<sup>(0)</sup> is then drawn from a normal distribution with mean 0.5 and standard deviation 0.25, truncated to (0, 1), so the interval spans ±2 standard deviations (`method.random_init`). `--judge-rounds V` likewise overrides the number of single-item judge rounds. Reports from runs with a non-default K or V are named `<dataset>_k<K>_v<V>_report.json`.

Every report records:

- **`judge_rounds`:** one row per round, including the Stage 2 row (round 0). Each row has the selected item, its information gain, the judge's answer and time, and the EM iterations after the round. It also has the iterations a cold start from θ<sup>(0)</sup> needs on the same data, the resulting iteration reduction per round and cumulatively, and the MAE after the round.
- **`latency_seconds`:** end-to-end latency split into retrieval, database instances, model generation or training, execution, the three stages, and judge time.

`--repeats N` adds the paper's repeated-run protocol. Each repeat samples a pool of every size in `experiments.pool_sizes` (5 to 30, capped at the pool size) from the full pool with a fixed seed. It reruns Stages 1–3 on the sampled models and scores them against their true accuracy. Node pools draw one of the three training seeds for each architecture. The report's `repeated` section gives the mean, SD, and 95% confidence interval of every metric and of latency, overall and per pool size, plus the per-round MAE and iteration-reduction curves. Judge votes are cached per item and candidate set, so repeats reuse earlier judge calls and still count their original time:

```bash
python -m pooleval.pipeline text2sql --dataset spider --target-items 150 --judge --repeats 10
python -m pooleval.pipeline text2sql --dataset spider --subset-budget 0 --repeats 10   # K=0 point
python -m pooleval.pipeline image --dataset usps --judge --judge-rounds 14 --repeats 10
```

Image and node targets run the same three stages:

```bash
python -m pooleval.pipeline image --dataset usps --judge
python -m pooleval.pipeline node --dataset dblpv7 --judge
python -m pooleval.pipeline node --dataset dblpv7 --skip-external   # PyG members only
```

The image pipeline samples `target_items` images from the target (1000 by default). It builds 50 meta-subsets of 200 source-test images, each with one synthetic shift family at a random severity (rotation, color, blur, noise, perspective, contrast, posterize, invert, sharpness, or occlusion). It retrieves the top `K=10` subsets by DINOv2 similarity, and the Qwen2.5-VL judge runs `V=6` rounds. ImageNet targets use each classifier's native ImageNet-1K head. Digit and CIFAR targets get a linear head per model, trained on `head_train_items` frozen features from the source training split. The CLIP and SigLIP models classify zero-shot from class-name prompts.

The node pipeline samples `target_items` target nodes. It builds 50 GNNEvaluator-style meta-graphs from the held-out labeled source nodes, rotating through EdgeDrop, node-feature masking, and subgraph sampling at random rates. It retrieves the top `K=12` meta-graphs by the mean of propagated node features, (D<sup>-1/2</sup>(A+I)D<sup>-1/2</sup>)<sup>2</sup>X. It trains each PyG architecture on the labeled source training nodes with three seeds (`tasks.node.train_seeds`); the first seed forms the main pool, and repeated runs draw one per architecture. GOFA runs `V=5` judge rounds.

### External graph runtimes

GraphGPT-7B, GraphPFN-1.3, and the two LLaGA checkpoints run inside their official code through the runners in [`scripts/runners/`](scripts/runners/):

| Member | Runner | Environment |
|---|---|---|
| GraphGPT-7B | `graphgpt_runner.py` | The environment of the cloned GraphGPT repository. Loads GraphGPT-7B-mix-all with the Arxiv-PubMed-GraphCLIP-GT graph tower, as in `graphgpt/eval/run_graphgpt.py`. Each target node is a center-first 2-hop subgraph with 128-dimensional node features (ogbn-arxiv's own, otherwise PCA to 128 dimensions) and its text. |
| GraphPFN-1.3 | `graphpfn_runner.py` | `pip install graphpfn` (PyTorch < 2.5 with DGL) and the LimiX backbone under its license. In-context prediction: the labeled source training nodes are the context, and other graphs are appended as disjoint components. |
| LLaGA-HO / LLaGA-ND | `llaga_runner.py --template HO\|ND` | The environment of the cloned LLaGA repository, with `sentence-transformers`. Node embeddings concatenate the three SimTeG encoders (2432 dimensions). ND adds LLaGA's 2-hop, 10-neighbor sequence and Laplacian encoding; HO uses 0–2-hop propagated embeddings, as in `eval/eval_pretrain.py`. |

`python -m pooleval.models download --task node` fetches the checkpoints, and `python -m pooleval.models runtimes --task node` clones the official repositories into `checkpoints/runtimes/`. Set `python` for each entry under `external_runners` in `configs/paper.yaml` to the interpreter of that model's environment:

```yaml
external_runners:
  GraphGPT-7B: {python: /envs/graphgpt/bin/python, script: scripts/runners/graphgpt_runner.py}
```

The pipeline calls `python script --job job.json --output predictions.json`. `job.json` contains:

- the model name, its downloaded `checkpoint`, `base`, and `auxiliary` paths, and its cloned `runtime` directory;
- the `class_names` and a `domain` description;
- `train`: the source training graph, its training node ids, and their labels;
- one entry per requested graph: `key`, `graph` (a torch file with `x` and `edge_index` only; labels of requested nodes are never written), `nodes`, `node_text` (a JSON list, or null), and `same_as_train`.

The runner writes `{key: [class index, or -1 when the answer maps to no class]}`. `--skip-external` leaves these members out and lists them under `excluded_models` in the report.

The paper configuration fixes the seed to `42`. At pipeline startup, `seed_everything()` seeds Python, NumPy, PyTorch, and every available CUDA device. It also requests deterministic PyTorch algorithms, disables cuDNN benchmarking, and configures deterministic cuDNN behavior. Use `--seed N` to override the configured seed. Dataset code that creates a PyTorch `DataLoader` should include the supplied worker seeding options:

```python
from pooleval.reproducibility import data_loader_seed_options
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    num_workers=4,
    **data_loader_seed_options(seed=42),
)
```

PoolEvaluator's closed-form EM runs in NumPy at float64 precision.

The Text2SQL execution path is:

1. Group labeled calibration examples by database, embed their questions with ColBERTv2, and retrieve the top `K=15` subsets by cosine similarity of the mean embeddings. With `K=0`, skip to step 2 and use the random truncated-normal prior in step 4.
2. Run each pool member lazily on calibration and target prompts.
3. Execute every SQL answer on the original database and five variants. Two answers agree only if their canonical results agree on all six instances.
4. Compute leave-one-model-out consensus and initialize α, β, and γ from the retrieved labeled subsets.
5. Run closed-form EM until the infinity-norm change is below `1e-6`.
6. For up to `V=10` rounds, select the unjudged item with the largest expected information gain (expected reduction in posterior entropy of the remaining items), ask the judge ensemble to choose among its executable candidates or NONE, hard-fix the revealed correctness vector, and warm-start EM. Stop early when no item has positive gain.
7. Write estimated accuracies, ranking, true held-out EX-Extended metrics, per-round logs, latency, and run metadata to `artifacts/<dataset>_report.json`.

Gold SQL is used only to compute calibration priors and post-hoc evaluation metrics. It is never included in a target model prompt or judge prompt.

## 7. Run everything from one script

The default is a safe smoke run: unit tests, all model-pool checks, a download plan, estimator inference, and one real Spider plus one real BIRD database-instance test.

```bash
bash scripts/run_all.sh
```

The full paper-scale Text2SQL run uses all 35 models, 150 target items per dataset, top-15 retrieved subsets, EX-Extended, and ten judge rounds:

```bash
MODE=full \
DOWNLOAD_MODELS=1 \
PREPARE_EXTERNAL_RUNTIMES=1 \
RUN_JUDGES=1 \
REQUIRE_ALL_JUDGES=1 \
bash scripts/run_all.sh
```

Useful controls:

```bash
TARGET_ITEMS=10          # target examples per dataset
SOURCE_CANDIDATES=500    # calibration records considered before top-K retrieval
MAX_MODELS=5             # prefix of the 35-member Text2SQL pool
INFERENCE_DTYPE=bfloat16 # use float32 automatically when bfloat16 is unsupported
SEED=42                  # Python, NumPy, PyTorch, CUDA, and loader seed
DOWNLOAD_MODELS=0        # reuse existing checkpoints
PREPARE_EXTERNAL_RUNTIMES=0 # clone official custom graph-model code
DOWNLOAD_SUPPORT=1       # download ColBERTv2, DINOv2, Qwen2.5-VL-72B, and GOFA
SUBSET_BUDGET=0          # override K for every task; 0 = random truncated-normal prior
JUDGE_ROUNDS=14          # override V for every task
REPEATS=10               # repeated runs over sampled pools with 95% CIs (0 = single run)
TEXT2SQL_TARGETS="spider bird spider2 beaver sciencebenchmark entsql livesqlbench"
RUN_JUDGES=0             # stop after Stage 2
RUN_IMAGE=1              # also run the image targets (IMAGE_TARGETS="usps svhn ...")
RUN_NODE=1               # also run the node targets (NODE_TARGETS="acmv9 dblpv7 ...")
SKIP_EXTERNAL=1          # node runs without GraphGPT/GraphPFN/LLaGA runners
LOAD_ALL_POOLS=1         # additionally load all Text2SQL/image/node members in sequence
DB_SMOKE_LIMIT=2         # databases per dataset in the initial database smoke test
```

The image and node runs need their datasets under `$POOLEVAL_DATASETS/image` and `$POOLEVAL_DATASETS/node` (section 2).

## 8. Tests and repository map

```bash
python -m pytest -q
```

```text
assets/                  paper figures used above
configs/paper.yaml       per-task K and V, EM, random prior, paths, and judge defaults
model_pools/             all appendix model pools
pooleval/estimator.py    leave-one-out agreement and closed-form EM
pooleval/retrieval.py    top-K Text2SQL subset retrieval
pooleval/benchmarks.py   Spider 2.0, BEAVER, ScienceBenchmark, EntSQL, and LiveSQLBench loaders
pooleval/sql_dialects.py PostgreSQL/MySQL execution and EX-Extended instances
pooleval/result_match.py Spider 2.0 and EntSQL result-table matchers
pooleval/repeats.py      repeated runs over sampled pools with 95% CIs
pooleval/encoders.py     ColBERTv2, DINOv2, and graph-propagation encoders
pooleval/databases.py    five-instance SQLite generator
pooleval/execution.py    read-only SQL and EX-Extended equivalence
pooleval/models.py       checkpoint download and lazy loading
pooleval/adapters.py     Text2SQL/image/node prediction adapters
pooleval/judges.py       Text2SQL API judge ensemble
pooleval/local_judges.py Qwen2.5-VL image judge and GOFA node judge
pooleval/core.py         shared Stage 1-3 driver
pooleval/pipeline.py     CLI and the Text2SQL pipeline
pooleval/image_data.py   image sources, targets, and synthetic-shift meta-subsets
pooleval/image_pipeline.py image-classification pipeline
pooleval/graph_data.py   graph sources, targets, and GNNEvaluator meta-graphs
pooleval/node_metadata.py node text and class names from public raw releases
pooleval/node_pipeline.py node-classification pipeline and external runners
pooleval/reproducibility.py runtime seeds and deterministic PyTorch configuration
scripts/run_all.sh       smoke and full reproduction entry point
scripts/runners/         GOFA judge server and GraphGPT/GraphPFN/LLaGA runners
tests/                   deterministic unit and integration tests
```
