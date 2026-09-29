PY = ".venv/bin/python"
PP = "PYTHONPATH=src"


rule all:
    input:
        "results/summary.json",
        "results/summary_aav.json",


rule fetch:
    output:
        "data/raw/gb1_flip.csv.zip",
    shell:
        "{PP} {PY} -c \"from protein_diffusion.data import fetch_raw; "
        "from protein_diffusion.config import load_config; "
        "fetch_raw(load_config()['dataset']['url'], '{output}')\""


rule prepare:
    input:
        rules.fetch.output,
    output:
        "data/processed/gb1.parquet",
    shell:
        "{PP} {PY} -m protein_diffusion.data {input} {output}"


rule run:
    input:
        rules.prepare.output,
    output:
        summary="results/summary.json",
        model="data/processed/ddpm.pt",
    shell:
        "{PP} {PY} -m protein_diffusion.run {input} {output.summary} {output.model}"


# --- second landscape: AAV2 capsid viability ---

AAV_CFG = "config/config_aav.yaml"


rule fetch_aav:
    output:
        "data/raw/aav_full.csv.zip",
    shell:
        "DIFFUSION_CONFIG={AAV_CFG} {PP} {PY} -c \""
        "from protein_diffusion.data import fetch_raw; "
        "from protein_diffusion.config import load_config; "
        "fetch_raw(load_config()['dataset']['url'], '{output}')\""


rule prepare_aav:
    input:
        rules.fetch_aav.output,
    output:
        "data/processed/aav.parquet",
    shell:
        "DIFFUSION_CONFIG={AAV_CFG} {PP} {PY} -m protein_diffusion.data {input} {output}"


rule run_aav:
    input:
        rules.prepare_aav.output,
    output:
        summary="results/summary_aav.json",
        model="data/processed/ddpm_aav.pt",
    shell:
        "DIFFUSION_CONFIG={AAV_CFG} {PP} {PY} -m protein_diffusion.run "
        "{input} {output.summary} {output.model}"


# --- scale-out: real 2-process DDP via torchrun (CPU gloo) ---

DDP_PROC = 2


rule train_ddp:
    input:
        rules.prepare.output,
    output:
        model="data/processed/ddpm_ddp.pt",
        record="results/train_ddp.json",
    shell:
        "{PP} .venv/bin/torchrun --nproc_per_node={DDP_PROC} "
        "-m protein_diffusion.train_ddp {input} {output.model} {output.record}"

rule eval_ddp:
    input:
        data=rules.prepare.output,
        model="data/processed/ddpm_ddp.pt",
    output:
        "results/eval_ddp.json",
    shell:
        "{PP} .venv/bin/python -m protein_diffusion.eval_ddp "
        "{input.data} {input.model} {output}"

rule diagnostics_aav:
    input:
        data=rules.prepare_aav.output,
        model=rules.run_aav.output.model,
    output:
        "results/diagnostics_aav.json",
    shell:
        "DIFFUSION_CONFIG={AAV_CFG} {PP} {PY} -m protein_diffusion.diagnostics "
        "{input.data} {input.model} {output}"

rule parity_jax:
    input:
        data=rules.prepare.output,
        model="data/processed/ddpm.pt",
    output:
        "results/parity_jax.json",
    shell:
        "{PP} {PY} -m protein_diffusion.eval_jax "
        "{input.data} {input.model} {output}"
