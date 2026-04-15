"""
Event Logger Module.
Provides timestamped logging of identity recognition events to both
console and file, with structured event tracking for dashboard use.
"""

import logging
import os
from datetime import datetime


class EventLogger:
    """Structured event logger for identity recognition system."""

    def __init__(self, log_file=None):
        if log_file is None:
            from utils.config import NOTIFICATION_LOG_FILE
            log_file = NOTIFICATION_LOG_FILE

        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Set up Python logging
        self.logger = logging.getLogger('IdentitySystem')
        self.logger.setLevel(logging.DEBUG)

        # Avoid duplicate handlers on re-initialization
        if not self.logger.handlers:
            # File handler — captures everything
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setLevel(logging.DEBUG)

            # Console handler — INFO and above
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-7s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

        # In-memory event store for dashboard access
        self.events = []

    def log_event(self, event_type, identity, confidence, message, **kwargs):
        """Log a structured recognition event.

        Args:
            event_type: One of 'INFO', 'WARNING', 'ALERT', 'CRITICAL'
            identity: Name of the identified person (or 'Unknown')
            confidence: Confidence score [0.0, 1.0]
            message: Human-readable notification message
            **kwargs: Additional metadata to attach to the event

        Returns:
            The event dictionary.
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'identity': identity,
            'confidence': round(confidence, 4),
            'message': message,
            **kwargs
        }
        self.events.append(event)

        # Map event types to Python log levels
        level_map = {
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ALERT': logging.WARNING,
            'CRITICAL': logging.CRITICAL
        }
        level = level_map.get(event_type, logging.INFO)
        self.logger.log(level, f"[{event_type}] {message} (confidence: {confidence:.2%})")

        return event

    def get_recent_events(self, n=10):
        """Get the N most recent events."""
        return self.events[-n:]

    def get_events_for_identity(self, identity):
        """Get all events for a specific identity."""
        return [e for e in self.events if e['identity'] == identity]

    def get_event_count(self):
        """Get total number of recorded events."""
        return len(self.events)
