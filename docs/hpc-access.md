# NYU Torch HPC — access notes

Quick reference for connecting to NYU Torch and the layout we use for Terra Perceive.

## SSH

Aliases live in `~/.ssh/config` (already configured):

| Alias | Host | Purpose |
|---|---|---|
| `torch` | `login.torch.hpc.nyu.edu` | interactive login — compile, sbatch, monitor |
| `dtn` | `dtn.torch.hpc.nyu.edu` | data-transfer node — rsync/scp of large datasets |

Auth is Microsoft device login (no SSH keys). `ControlMaster` keeps the master socket alive ~24h after last activity so MFA only fires once per host.

```bash
ssh torch                # interactive
ssh dtn                  # bulk data
ssh -O check torch       # is the master socket alive?
ssh -O exit torch        # force re-auth
```

User: `np3129`.

## On-HPC layout

```
/scratch/np3129/
├── terra-perceive-p2m4/         # repo (rsynced from laptop)
│   ├── data/
│   │   └── RELLIS-3D -> /scratch/np3129/data/RELLIS-3D   (symlink)
│   ├── third_party -> /scratch/np3129/third_party        (symlink)
│   └── ...
├── data/RELLIS-3D/              # bag files (5 × ~6 GB)
│   ├── 00000_00.bag
│   ├── 00000_01.bag
│   ├── 00000_02.bag
│   ├── 00000_03.bag
│   └── 00000_04.bag
├── conda_envs/terra_perceive_m4/   # conda prefix env
├── conda_pkgs/                  # cache
├── third_party/                 # tinycolormap, stb headers
└── m4_perframe/                 # ablation outputs
```

Why `$SCRATCH` for everything: home quota is small; `$SCRATCH` is the only place big enough for bags + build artifacts + conda envs. `$SCRATCH` is *not* backed up — treat it as cache.

## Sync flows

```bash
# Laptop → HPC (code + bags). Bags use rsync --partial; resumable.
bash scripts/sync_to_hpc.sh

# HPC → laptop (results, plots, logs).
bash scripts/sync_from_hpc.sh
```

The sync script self-heals a broken `data/` entry on HPC (file or dangling symlink instead of dir).

## One-time setup

```bash
ssh torch
cd /scratch/$USER/terra-perceive-p2m4
bash scripts/setup_hpc_p2m4.sh   # builds conda prefix env, links third_party
```

## Conda env

```bash
source /scratch/np3129/conda_envs/terra_perceive_m4/etc/profile.d/conda.sh
conda activate /scratch/np3129/conda_envs/terra_perceive_m4
```

## Slurm

```bash
sbatch slurm/run_ablation_g.slurm
squeue -u $USER
scancel <jobid>
tail -f /scratch/$USER/p2m4_logs/<jobid>.out
```

## Repairing a broken `data/RELLIS-3D` symlink (manual)

If the sync script's auto-repair ever fails, run on HPC:

```bash
cd /scratch/$USER/terra-perceive-p2m4
ls -la data 2>/dev/null            # diagnose: file? broken symlink? dir?
[ -e data ] && [ ! -d data ] && rm -f data
[ -L data ] && [ ! -e data ] && rm -f data
mkdir -p data
[ -L data/RELLIS-3D ] && [ ! -e data/RELLIS-3D ] && rm -f data/RELLIS-3D
[ ! -L data/RELLIS-3D ] && ln -s /scratch/$USER/data/RELLIS-3D data/RELLIS-3D
ls -la data/
```

Verify: `ls data/RELLIS-3D/*.bag` should list the 5 bags.
