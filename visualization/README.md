# Live CSV Bar Chart

A local web app with a nice animated bar chart that **updates instantly** whenever you edit `data.csv` — no refresh needed.

## Run it

```bash
pip install -r requirements.txt
python server.py
```

Then open **http://localhost:5000** in your browser.

## Try the live update

While the server is running, open `data.csv` in any editor (Excel, Notepad, VS Code...), change or add a row, and hit save. The chart in the browser will animate to the new values within a fraction of a second — no manual refresh.

`data.csv` must have exactly two columns: `word,count`.

## How it works

- `server.py` — a Flask + Flask-SocketIO app. A `watchdog` filesystem observer watches `data.csv`; any save triggers it to re-read the CSV and push the new data over a WebSocket to every connected browser.
- `templates/index.html` — the frontend: a Chart.js animated bar chart with a dark glassmorphism UI, live "connected" status dot, and summary stat cards (entry count, total, top word, last updated time).

## Notes

- Rows are automatically sorted by count, descending.
- If the CSV is malformed (missing columns, bad data), the page shows a red error banner instead of crashing.
- To use a different CSV file, change `CSV_PATH` at the top of `server.py`.
