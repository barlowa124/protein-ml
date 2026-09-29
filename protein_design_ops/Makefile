.PHONY: test run dry

test:
	.venv/bin/python -m pytest tests/ -q

run:
	.venv/bin/python -m snakemake --cores 2

dry:
	.venv/bin/python -m snakemake -n
