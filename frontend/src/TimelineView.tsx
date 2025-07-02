import React, { useEffect, useState } from 'react';
import { fetchTimeline, TimelineEvent } from './api';

// TimelineView: Displays the event timeline
// Future: Fetch events from backend, render timeline, add filtering/search

const TimelineView: React.FC = () => {
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchTimeline().then(data => {
      setEvents(data);
      setLoading(false);
    });
  }, []);

  if (loading) return <div>Loading timeline...</div>;

  return (
    <div>
      <h2>Event Timeline</h2>
      <ul>
        {events.map(event => (
          <li key={event.id}>
            <strong>{event.event_time}</strong> [{event.ticker}] {event.summary} <em>({event.tags.join(', ')})</em>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default TimelineView; 