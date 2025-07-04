# Frontend Backend Integration Updates

This document outlines the changes made to the frontend to integrate with the updated backend infrastructure.

## 🔄 **Backend Infrastructure Changes Detected:**

### **1. Database Structure**
- **PostgreSQL** with `events` and `agent_status` tables
- **JSONB storage** for flexible event data
- **Proper indexing** for fast timeline queries

### **2. API Endpoints (Orchestrator)**
- `GET /health` - Health check
- `GET /timeline` - Fetch timeline events  
- `GET /agents/status` - Fetch agent status
- `POST /query` - Send user queries
- `POST /mcp` - Agent-to-agent communication

### **3. Microservice Architecture**
- **13 agent services** running as Docker containers
- **Centralized orchestrator** for coordination
- **Environment-based configuration**

## 📝 **Frontend Changes Made:**

### **1. Updated API Service (`src/api.ts`)**
- **Proper error handling** with HTTP status checks
- **Backend health checks** before making requests
- **Updated data types** to match database schema
- **Fallback mock data** when backend is not ready
- **Utility functions** for timestamp formatting

### **2. Enhanced Components**
- **AgentStatusPanel**: Real-time agent monitoring with health indicators
- **TimelineView**: Beautiful event timeline with proper formatting
- **Backend status indicators** in UI components

### **3. Configuration Management**
- **Environment variables** for API endpoints and settings
- **Configurable polling intervals** for real-time updates
- **Backend health monitoring** integration

### **4. Data Structure Updates**
```typescript
// Updated to match backend database
interface TimelineEvent {
  id: number;
  event_time: string;  // ISO timestamp
  source_agent: string;
  event_type: string;
  ticker?: string;
  data: any;          // JSONB data from backend
  tags: string[];
  summary?: string;   // Generated for display
  created_at?: string;
}

interface AgentStatus {
  id?: number;
  agent_name: string;
  last_run?: string;  // ISO timestamp
  status: string;
  details?: any;      // JSONB details
}
```

## 🚀 **Setup Instructions:**

### **1. Install Dependencies**
```bash
cd frontend
npm install @types/node @types/react @types/react-dom react react-dom react-scripts typescript recharts framer-motion styled-components @types/styled-components lucide-react react-countup
```

### **2. Environment Configuration**
```bash
# Copy environment file
cp .env.example .env

# Update .env with your backend URL
REACT_APP_API_URL=http://localhost:8000
```

### **3. Start Development**
```bash
npm start
```

## 🔧 **Backend Integration Features:**

### **1. Real-time Updates**
- **Automatic polling** every 30 seconds for timeline and agent status
- **Configurable intervals** via environment variables
- **Error handling** with fallback to mock data

### **2. Health Monitoring**
- **Backend health checks** before API calls
- **Visual indicators** showing backend status
- **Graceful degradation** when backend is offline

### **3. Data Transformation**
- **Automatic summary generation** from JSONB event data
- **Timestamp formatting** for better UX
- **Agent name formatting** (snake_case → Title Case)

### **4. Error Handling**
- **Comprehensive error logging** for debugging
- **User-friendly error messages** 
- **Fallback mock data** for development

## 🔮 **Future Backend Endpoints to Implement:**

When these endpoints are ready, update the frontend:

```typescript
// Trading data endpoints
GET /trading/stats        // Trading statistics
GET /trading/pnl         // P&L chart data  
GET /trading/positions   // Current positions
GET /system/status       // System status

// Real-time updates
WebSocket /ws            // Real-time data stream
```

## 🐛 **Testing Backend Integration:**

### **1. With Backend Running**
```bash
# Start the full stack
docker-compose up

# Frontend should connect to real data
npm start
```

### **2. Without Backend (Mock Mode)**
```bash
# Set environment variable
REACT_APP_USE_MOCK_DATA=true

# Frontend will use mock data
npm start
```

### **3. Health Check**
The frontend will automatically:
- ✅ Check `/health` endpoint
- ✅ Show "Backend Online/Offline" indicator
- ✅ Fall back to mock data if needed

## 📊 **Data Flow:**

```
Frontend → Orchestrator → Database
    ↓          ↓           ↓
Components → FastAPI → PostgreSQL
    ↓          ↓           ↓
 Real-time ← Events ← Agent Data
```

## ⚙️ **Configuration Options:**

### **Environment Variables**
- `REACT_APP_API_URL` - Backend orchestrator URL
- `REACT_APP_USE_MOCK_DATA` - Enable/disable mock data
- `REACT_APP_*_REFRESH_INTERVAL` - Polling intervals

### **Runtime Configuration**
- Automatic backend detection
- Graceful fallback to mock data
- Real-time polling with configurable intervals

---

The frontend is now fully prepared to work with your updated backend infrastructure! 🎯