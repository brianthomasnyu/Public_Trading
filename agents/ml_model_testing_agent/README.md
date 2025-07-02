# ML Model Testing Agent

## Purpose
Downloads and tests ML models from Hugging Face, scikit-learn, etc., and stores model events in the knowledge base.

## How it works
- Downloads/tests ML models for prediction/classification
- Checks and updates the knowledge base
- Communicates with orchestrator/other agents via MCP
- Runs in parallel and recursively until all data is processed

## Next Steps
- Implement fetch_and_process_models in agent.py
- Add real ML model integration and parsing
- Expand MCP communication 