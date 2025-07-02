import React from 'react';
import QueryBar from './QueryBar';
import TimelineView from './TimelineView';
import AgentStatusPanel from './AgentStatusPanel';

// Main App component: renders core UI components
const App: React.FC = () => {
  return (
    <div>
      <h1>Public Trading AI Frontend</h1>
      <QueryBar />
      <TimelineView />
      <AgentStatusPanel />
      {/* Future: Add DocumentUploader, KPIView, SearchBar, etc. */}
    </div>
  );
};

export default App; 