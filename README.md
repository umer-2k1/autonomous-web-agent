# Autonomous Web Agent

Give it a goal in plain English and it drives a real browser to carry it out, navigating pages, reading them, and taking screenshots as it goes.

## How it works
A Groq-powered ReAct agent (LangGraph) controls Chromium through Playwright. Each browser action (navigate, screenshot, interact) is exposed to the agent as a tool, so it plans and acts step by step instead of following a fixed script.

## Run
```bash
pip install -r requirements.txt
playwright install chromium
# set GROQ_API_KEY
python main.py
```
