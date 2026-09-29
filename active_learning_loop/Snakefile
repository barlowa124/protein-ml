PY = ".venv/bin/python"
PP = "PYTHONPATH=src"


rule all:
    input:
        "results/summary.json",
        "results/curves.png",
        "results/summary_aav.json",
        "results/curves_aav.png",
        "results/summary_gb1_esm2.json",
        "results/curves_gb1_esm2.png",
        "results/summary_aav_esm2.json",
        "results/curves_aav_esm2.png",


rule fetch:
    output:
        "data/raw/gb1_flip.csv.zip",
    shell:
        "{PP} {PY} -c \"from al_loop.data import fetch_raw; "
        "from al_loop.config import load_config; "
        "fetch_raw(load_config()['dataset']['url'], '{output}')\""


rule prepare:
    input:
        rules.fetch.output,
    output:
        "data/processed/gb1.parquet",
    shell:
        "{PP} {PY} -m al_loop.data {input} {output}"


rule run:
    input:
        rules.prepare.output,
    output:
        records="data/processed/records.parquet",
        picks="data/processed/active_picks.parquet",
    shell:
        "{PP} {PY} -m al_loop.loop {input} {output.records} {output.picks}"


rule report:
    input:
        records=rules.run.output.records,
        landscape=rules.prepare.output,
    output:
        "results/summary.json",
        "results/curves.png",
    shell:
        "{PP} {PY} -m al_loop.evaluate "
        "{input.records} {input.landscape} {output[0]} {output[1]}"


# --- second landscape: AAV2 capsid viability (replication, not re-tuning) ---
# Identical experiment schedule; only the dataset descriptor differs.

AAV_CFG = "config/config_aav.yaml"


rule fetch_aav:
    output:
        "data/raw/aav_full.csv.zip",
    shell:
        "AL_CONFIG={AAV_CFG} {PP} {PY} -c \"from al_loop.data import fetch_raw; "
        "from al_loop.config import load_config; "
        "fetch_raw(load_config()['dataset']['url'], '{output}')\""


rule prepare_aav:
    input:
        rules.fetch_aav.output,
    output:
        "data/processed/aav.parquet",
    shell:
        "AL_CONFIG={AAV_CFG} {PP} {PY} -m al_loop.data {input} {output}"


rule run_aav:
    input:
        rules.prepare_aav.output,
    output:
        records="data/processed/records_aav.parquet",
        picks="data/processed/active_picks_aav.parquet",
    shell:
        "AL_CONFIG={AAV_CFG} {PP} {PY} -m al_loop.loop "
        "{input} {output.records} {output.picks}"


rule report_aav:
    input:
        records=rules.run_aav.output.records,
        landscape=rules.prepare_aav.output,
    output:
        "results/summary_aav.json",
        "results/curves_aav.png",
    shell:
        "AL_CONFIG={AAV_CFG} {PP} {PY} -m al_loop.evaluate "
        "{input.records} {input.landscape} {output[0]} {output[1]}"


# --- encoder ablation on GB1: ESM-2 embeddings in WT context vs one-hot ---

ESM_CFG = "config/config_gb1_esm2.yaml"


rule run_gb1_esm2:
    input:
        rules.prepare.output,
    output:
        records="data/processed/records_gb1_esm2.parquet",
        picks="data/processed/active_picks_gb1_esm2.parquet",
    shell:
        "AL_CONFIG={ESM_CFG} {PP} {PY} -m al_loop.loop "
        "{input} {output.records} {output.picks}"


rule report_gb1_esm2:
    input:
        records=rules.run_gb1_esm2.output.records,
        landscape=rules.prepare.output,
    output:
        "results/summary_gb1_esm2.json",
        "results/curves_gb1_esm2.png",
    shell:
        "AL_CONFIG={ESM_CFG} {PP} {PY} -m al_loop.evaluate "
        "{input.records} {input.landscape} {output[0]} {output[1]}"


# --- encoder ablation on AAV2: the case where embeddings should help ---

AAV_ESM_CFG = "config/config_aav_esm2.yaml"


rule run_aav_esm2:
    input:
        rules.prepare_aav.output,
    output:
        records="data/processed/records_aav_esm2.parquet",
        picks="data/processed/active_picks_aav_esm2.parquet",
    shell:
        "AL_CONFIG={AAV_ESM_CFG} {PP} {PY} -m al_loop.loop "
        "{input} {output.records} {output.picks}"


rule report_aav_esm2:
    input:
        records=rules.run_aav_esm2.output.records,
        landscape=rules.prepare_aav.output,
    output:
        "results/summary_aav_esm2.json",
        "results/curves_aav_esm2.png",
    shell:
        "AL_CONFIG={AAV_ESM_CFG} {PP} {PY} -m al_loop.evaluate "
        "{input.records} {input.landscape} {output[0]} {output[1]}"
