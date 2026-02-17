.PHONY: setup sanity

setup:
    pip install -r requirements.txt

sanity:
    python -m app.sanity
    @echo "-> artifacts/sanity_output.json written"
