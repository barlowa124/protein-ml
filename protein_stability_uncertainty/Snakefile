PY = ".venv/bin/python"
PP = "PYTHONPATH=src"


rule all:
    input:
        "results/summary.json",


rule fetch:
    output:
        "data/raw/meltome_splits.zip",
    shell:
        "{PP} {PY} -c \"from protstab.data import fetch_raw; "
        "from protstab.config import load_config; "
        "fetch_raw(load_config()['dataset']['url'], '{output}')\""


rule prepare:
    input:
        rules.fetch.output,
    output:
        "data/processed/meltome.parquet",
    shell:
        "{PP} {PY} -m protstab.data {input} {output}"


rule run:
    input:
        rules.prepare.output,
    output:
        "results/summary.json",
        "results/deploy_composition.json",
        "results/deploy_esm2.json",
    shell:
        "{PP} {PY} -m protstab.run {input} {output[0]}"
