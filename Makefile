.PHONY: help install pipeline test app reports notebook

help:
	@echo "Card Fraud Triage Project - Make Commands:"
	@echo "  make install    - Install required dependencies"
	@echo "  make pipeline   - Run full end-to-end experiment pipeline"
	@echo "  make test       - Run full unit and integration test suite"
	@echo "  make app        - Launch local Streamlit interactive triage dashboard"
	@echo "  make reports    - Generate Phase 1 & Phase 2 IEEE PDF reports"
	@echo "  make notebook   - Execute all 11 modular research notebooks"

install:
	pip install -r requirements.txt

pipeline:
	python3 src/pipeline.py

test:
	python3 -m unittest discover tests

app:
	streamlit run app/app.py

reports:
	python3 scripts/generate_pdf_reports.py

notebook:
	python3 scripts/execute_all_notebooks.py
