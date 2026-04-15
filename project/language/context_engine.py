"""
Context Engine Module.
Tracks temporal context for detected persons — when they were first/last seen,
visit frequency, session statistics — enabling intelligent notification generation.
"""

from datetime import datetime, timedelta


class ContextEngine:
    """Temporal context tracker for identity recognition sessions."""

    def __init__(self):
        self.tracking = {}       # person_id → list of sighting records
        self.session_start = datetime.now()
        self.unknown_count = 0
        self.total_detections = 0

    def update(self, fusion_result):
        """Update tracking with a new detection from the fusion engine.

        Args:
            fusion_result: Dict from MultimodalFusion.fuse()

        Returns:
            Context dict enriched with temporal information
        """
        self.total_detections += 1
        now = datetime.now()

        person_id = fusion_result.get('person_id')
        identity = fusion_result.get('identity', 'Unknown')
        confidence = fusion_result.get('confidence', 0)
        status = fusion_result.get('status', 'UNKNOWN')

        # Handle unknown persons
        if identity == 'Unknown' or person_id is None:
            self.unknown_count += 1
            return self._build_context(
                identity=identity,
                person_id=None,
                is_new=True,
                is_returning=False,
                visit_count=0,
                duration=timedelta(0),
                last_seen=None,
                first_seen=None,
                status=status,
                confidence=confidence,
                timestamp=now
            )

        # Track known person
        if person_id not in self.tracking:
            self.tracking[person_id] = []

        sighting = {
            'timestamp': now,
            'confidence': confidence,
            'status': status
        }
        self.tracking[person_id].append(sighting)

        # Compute temporal context
        sightings = self.tracking[person_id]
        visit_count = len(sightings)
        first_seen = sightings[0]['timestamp']
        last_seen = sightings[-2]['timestamp'] if len(sightings) > 1 else None
        duration = now - first_seen

        # A "return visit" is defined as a gap > 30 seconds since last sighting
        is_returning = False
        if last_seen and (now - last_seen) > timedelta(seconds=30):
            is_returning = True

        return self._build_context(
            identity=identity,
            person_id=person_id,
            is_new=(visit_count == 1),
            is_returning=is_returning,
            visit_count=visit_count,
            duration=duration,
            last_seen=last_seen,
            first_seen=first_seen,
            status=status,
            confidence=confidence,
            timestamp=now
        )

    def _build_context(self, **kwargs):
        """Assemble a full context dictionary from keyword arguments."""
        return {
            'identity': kwargs.get('identity', 'Unknown'),
            'person_id': kwargs.get('person_id'),
            'is_new': kwargs.get('is_new', True),
            'is_returning': kwargs.get('is_returning', False),
            'visit_count': kwargs.get('visit_count', 0),
            'duration_in_view': kwargs.get('duration', timedelta(0)),
            'last_seen': kwargs.get('last_seen'),
            'first_seen': kwargs.get('first_seen'),
            'status': kwargs.get('status', 'UNKNOWN'),
            'confidence': kwargs.get('confidence', 0),
            'timestamp': kwargs.get('timestamp', datetime.now()),
            'session_total_detections': self.total_detections,
            'session_unknown_count': self.unknown_count,
            'time_of_day': self._get_time_of_day()
        }

    def _get_time_of_day(self):
        """Get a human-readable time-of-day label."""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'

    def get_person_history(self, person_id):
        """Get the full sighting history for a person."""
        return self.tracking.get(person_id, [])

    def get_active_persons(self, window_seconds=60):
        """Get persons detected within the last N seconds.

        Returns:
            Dict of person_id → detection count
        """
        now = datetime.now()
        cutoff = now - timedelta(seconds=window_seconds)
        active = {}
        for pid, sightings in self.tracking.items():
            recent = [s for s in sightings if s['timestamp'] > cutoff]
            if recent:
                active[pid] = len(recent)
        return active

    def get_session_summary(self):
        """Get an overview of the current recognition session."""
        return {
            'session_duration': str(datetime.now() - self.session_start),
            'total_detections': self.total_detections,
            'unique_persons': len(self.tracking),
            'unknown_encounters': self.unknown_count,
            'active_persons': self.get_active_persons()
        }
