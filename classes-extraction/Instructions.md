# CLI
python vocab_cli.py extract voters_2026.csv --column company_name -o words.csv
python vocab_cli.py edit words.csv -o words.json   # interactive prompts

# UI
streamlit run vocab_app.py