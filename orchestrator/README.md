# Unified AI Financial Data Orchestrator

## Overview

The Unified Orchestrator is the central coordination hub for the AI-powered financial data aggregation framework. It integrates multiple advanced AI tools to provide intelligent agent orchestration, dynamic tool selection, and sophisticated data processing capabilities.

## Architecture

### Multi-Tool Integration

The orchestrator integrates five key AI frameworks:

1. **LangChain** - Agent orchestration, memory management, and tool execution
2. **Computer Use** - Dynamic tool selection and self-healing capabilities
3. **LlamaIndex** - Advanced RAG (Retrieval-Augmented Generation) and knowledge base management
4. **Haystack** - Document analysis and question-answering
5. **AutoGen** - Multi-agent coordination and complex task decomposition

### Core Components

#### UnifiedOrchestrator Class
- **Agent Registry**: All 21 financial analysis agents registered as LangChain tools
- **Tool Selection**: Computer Use for intelligent tool selection based on query context
- **Memory Management**: LangChain conversation memory for context persistence
- **RAG Integration**: LlamaIndex for knowledge base queries and document retrieval
- **QA Pipeline**: Haystack for document-based question answering
- **Multi-Agent Coordination**: AutoGen for complex workflows requiring multiple agents

#### Enhanced Query Processing
1. **Query Classification**: Intent analysis with LangChain context integration
2. **Tool Selection**: Computer Use selects optimal agents for the query
3. **Parallel Execution**: LangChain agent executor orchestrates tool execution
4. **Knowledge Retrieval**: LlamaIndex provides relevant context from knowledge base
5. **Document Analysis**: Haystack processes documents for detailed answers
6. **Multi-Agent Workflows**: AutoGen coordinates complex tasks across multiple agents
7. **Result Aggregation**: Combines results from all sources with validation
8. **Memory Update**: Stores context for future queries

## Key Features

### Intelligent Agent Orchestration
- **Dynamic Tool Selection**: Computer Use automatically selects the best agents for each query
- **Context Awareness**: LangChain memory maintains conversation context
- **Parallel Processing**: Multiple agents can work simultaneously on different aspects
- **Self-Healing**: Automatic error recovery and tool switching

### Advanced Data Processing
- **RAG Integration**: LlamaIndex provides intelligent document retrieval
- **Document QA**: Haystack extracts precise answers from documents
- **Multi-Agent Reasoning**: AutoGen enables complex reasoning chains
- **Knowledge Base Updates**: Continuous learning and knowledge accumulation

### Performance Monitoring
- **Agent Utilization Tracking**: Monitor which agents are most effective
- **Processing Time Analysis**: Track performance metrics
- **Error Rate Monitoring**: Identify and resolve issues
- **Memory Effectiveness**: Measure context retention and reuse

## API Endpoints

### Core Endpoints
- `POST /query` - Process financial analysis queries
- `GET /agents/status` - Get comprehensive agent status
- `GET /health` - System health check with component status

### LangChain Integration
- `POST /langchain/message` - Handle LangChain communication
- `GET /langchain/memory` - Access conversation memory
- `POST /langchain/test` - Test LangChain integration

### Enhanced Features
- `GET /timeline` - Timeline with LangChain context
- `GET /system/optimization` - Performance optimization suggestions
- `POST /system/test` - Integration testing

## Agent Tools

All 21 financial analysis agents are registered as LangChain tools:

1. **SEC Filings Agent** - Financial statement analysis
2. **Market News Agent** - News sentiment and media coverage
3. **Social Media NLP Agent** - Social sentiment analysis
4. **Equity Research Agent** - Analyst reports and ratings
5. **Insider Trading Agent** - Form 4 and insider activity
6. **Fundamental Pricing Agent** - Valuation analysis
7. **KPI Tracker Agent** - Performance metrics
8. **Event Impact Agent** - Catalyst analysis
9. **Options Flow Agent** - Options trading patterns
10. **Macro Calendar Agent** - Economic events
11. **Revenue Geography Agent** - Geographic analysis
12. **Data Tagging Agent** - Data organization
13. **Investor Portfolio Agent** - Institutional holdings
14. **Dark Pool Agent** - Alternative trading venues
15. **Short Interest Agent** - Short selling analysis
16. **Commodity Agent** - Commodity price monitoring
17. **ML Model Testing Agent** - Model validation
18. **Discovery Agent** - Question generation
19. **Repository Management Agent** - Code management
20. **API Key Management Agent** - Security management
21. **Comparative Analysis Agent** - Peer analysis

## Configuration

### Environment Variables
```bash
# Database Configuration
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=financial_data

# LangChain Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langchain_key
LANGCHAIN_PROJECT=financial_orchestrator

# OpenAI Configuration
OPENAI_API_KEY=your_openai_key

# Computer Use Configuration
COMPUTER_USE_API_KEY=your_computer_use_key

# LlamaIndex Configuration
LLAMA_INDEX_API_KEY=your_llama_index_key

# Haystack Configuration
HAYSTACK_API_KEY=your_haystack_key

# AutoGen Configuration
AUTOGEN_API_KEY=your_autogen_key
```

### Dependencies
See `requirements.txt` for complete dependency list including:
- `langchain` - Agent orchestration
- `computer-use` - Dynamic tool selection
- `llama-index` - RAG and knowledge base
- `haystack-ai` - Document QA
- `pyautogen` - Multi-agent coordination

## Usage Examples

### Basic Query Processing
```python
import requests

# Process a financial analysis query
response = requests.post("http://localhost:8000/query", json={
    "query": "Analyze Apple's Q4 earnings and compare to analyst expectations",
    "ticker": "AAPL",
    "priority": "high"
})

print(response.json())
```

### LangChain Integration
```python
# Send LangChain message
message = {
    "sender": "market_news_agent",
    "recipient": "orchestrator",
    "message_type": "tool_call",
    "content": {"query": "Get latest news sentiment for AAPL"},
    "langchain_trace_id": "trace_123"
}

response = requests.post("http://localhost:8000/langchain/message", json=message)
```

## Development

### Running the Orchestrator
```bash
# Start the orchestrator
python orchestrator/main.py

# Or using uvicorn
uvicorn orchestrator.main:app --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Run integration tests
curl -X POST http://localhost:8000/system/test

# Test LangChain integration
curl -X POST http://localhost:8000/langchain/test
```

## Monitoring and Analytics

### Performance Metrics
- Query processing time
- Agent utilization rates
- Tool selection patterns
- Error rates and recovery success
- Memory usage and context effectiveness

### Health Monitoring
- Agent availability and health
- LangChain component status
- Database connectivity
- API endpoint responsiveness

## Security and Compliance

### Critical Policy
**NO TRADING DECISIONS**: This orchestrator and all agents are strictly for data aggregation, analysis, and knowledge base management. No trading decisions should be made by any agent or the orchestrator.

### Data Protection
- Secure API key management
- Encrypted data transmission
- Access control and authentication
- Audit logging for all operations

## Troubleshooting

### Common Issues
1. **LangChain Memory Issues**: Check memory configuration and API keys
2. **Tool Selection Failures**: Verify Computer Use integration
3. **Agent Communication Errors**: Check network connectivity and agent health
4. **Performance Degradation**: Monitor resource usage and optimize queries

### Debugging
- Enable detailed logging
- Check LangChain tracing
- Monitor agent status endpoints
- Review performance metrics

## Future Enhancements

### Planned Features
- Advanced reasoning chains with AutoGen
- Enhanced RAG capabilities with LlamaIndex
- Improved tool selection algorithms
- Real-time streaming responses
- Advanced analytics dashboard

### Integration Roadmap
- Additional AI frameworks
- Enhanced security features
- Performance optimizations
- Extended agent capabilities

## Contributing

When contributing to the orchestrator:
1. Follow the existing code structure
2. Maintain the NO TRADING DECISIONS policy
3. Add comprehensive tests
4. Update documentation
5. Follow security best practices

## License

This project is licensed under the MIT License - see the LICENSE file for details. 