# Orchestrator Service
# --------------------
# This service coordinates all agent tasks, manages agent-to-agent (A2A) communication via MCP,
# receives user queries, dispatches tasks to agents, aggregates results, and updates the knowledge base.
# It will expose an API for the frontend and for agent communication.
# Future implementation: FastAPI app with endpoints for agent/task management and MCP message routing. 