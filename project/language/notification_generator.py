"""
Notification Generator Module.
Generates intelligent, context-aware human-readable notifications
using rule-based template logic that simulates LLM-style reasoning.
Produces natural language alerts for: known persons, unknown persons,
spoof attempts, modality conflicts, and returning visitors.
"""

from datetime import datetime


class NotificationGenerator:
    """Rule-based notification engine simulating LLM-style context-aware alerts."""

    def __init__(self):
        self.history = []

    def generate(self, context, fusion_result):
        """Generate a context-aware notification for a detection event.

        Args:
            context: Dict from ContextEngine.update()
            fusion_result: Dict from MultimodalFusion.fuse()

        Returns:
            Notification dict with: timestamp, identity, confidence,
            status, level, message, detail, icon
        """
        status = fusion_result.get('status', 'UNKNOWN')
        identity = context.get('identity', 'Unknown')
        confidence = context.get('confidence', 0)
        timestamp = context.get('timestamp', datetime.now())
        time_str = timestamp.strftime('%I:%M:%S %p')
        time_of_day = context.get('time_of_day', 'day')

        notification = {
            'timestamp': timestamp.isoformat(),
            'time_display': time_str,
            'identity': identity,
            'confidence': confidence,
            'status': status,
            'level': 'INFO',
            'message': '',
            'detail': '',
            'icon': '\U0001f464'  # 👤
        }

        # Route to appropriate notification generator
        if status == 'SPOOF':
            notification.update(
                self._spoof_notification(context, fusion_result, time_str)
            )
        elif status == 'CONFLICT':
            notification.update(
                self._conflict_notification(context, fusion_result, time_str)
            )
        elif status == 'CONFIRMED':
            notification.update(
                self._confirmed_notification(context, fusion_result, time_str, time_of_day)
            )
        elif status == 'IDENTIFIED':
            notification.update(
                self._identified_notification(context, fusion_result, time_str, time_of_day)
            )
        elif status == 'UNKNOWN':
            notification.update(
                self._unknown_notification(context, fusion_result, time_str)
            )

        self.history.append(notification)
        return notification

    # ──────────────────────────────────────────────────────────────
    # Notification type generators
    # ──────────────────────────────────────────────────────────────

    def _spoof_notification(self, context, fusion_result, time_str):
        """Generate alert for detected spoof attempt."""
        spoof_info = fusion_result.get('details', {}).get('spoof', {})
        reasons = spoof_info.get('reasons', ['Anomalous input detected'])
        reason_text = '; '.join(reasons)

        return {
            'level': 'CRITICAL',
            'icon': '\U0001f6a8',  # 🚨
            'message': f'SPOOF ALERT: Possible spoofing attempt detected at {time_str}',
            'detail': (
                f'Security alert - {reason_text}. '
                f'Spoof confidence: {spoof_info.get("spoof_confidence", 0):.0%}. '
                f'Recommend manual verification of the individual.'
            )
        }

    def _conflict_notification(self, context, fusion_result, time_str):
        """Generate warning for face/voice identity conflict."""
        details = fusion_result.get('details', {})
        conflict_info = details.get('conflict_info', {})
        face_says = conflict_info.get('face_says', 'Unknown')
        voice_says = conflict_info.get('voice_says', 'Unknown')
        primary = conflict_info.get('primary_modality', 'face')

        return {
            'level': 'WARNING',
            'icon': '\u26a0\ufe0f',  # ⚠️
            'message': f'Identity conflict detected at {time_str}',
            'detail': (
                f'Face recognition suggests "{face_says}" while voice recognition '
                f'suggests "{voice_says}". Using {primary} as primary source. '
                f'Overall confidence reduced to {context["confidence"]:.0%}. '
                f'Additional verification recommended.'
            )
        }

    def _confirmed_notification(self, context, fusion_result, time_str, time_of_day):
        """Generate notification for face+voice confirmed identity."""
        identity = context['identity']
        confidence = context['confidence']
        visit_count = context.get('visit_count', 0)
        is_returning = context.get('is_returning', False)
        greeting = self._get_greeting(time_of_day)

        if is_returning:
            duration_since = ""
            if context.get('last_seen'):
                diff = context['timestamp'] - context['last_seen']
                minutes = diff.total_seconds() / 60
                if minutes < 60:
                    duration_since = f" (last seen {minutes:.0f} minutes ago)"
                else:
                    duration_since = f" (last seen {minutes / 60:.1f} hours ago)"

            return {
                'level': 'INFO',
                'icon': '\U0001f44b',  # 👋
                'message': f'{identity} has returned at {time_str}{duration_since}',
                'detail': (
                    f'{greeting} {identity} (returning visitor, visit #{visit_count}). '
                    f'Identity confirmed via face + voice with {confidence:.0%} confidence.'
                )
            }
        elif context.get('is_new', False):
            return {
                'level': 'INFO',
                'icon': '\u2705',  # ✅
                'message': f'Known person detected: {identity} entered at {time_str}',
                'detail': (
                    f'{greeting} {identity}! Identity confirmed via both face and voice '
                    f'recognition with {confidence:.0%} confidence. First detection this session.'
                )
            }
        else:
            return {
                'level': 'INFO',
                'icon': '\U0001f464',  # 👤
                'message': f'{identity} is present (confirmed) - {time_str}',
                'detail': (
                    f'{identity} continues to be detected. Visit #{visit_count}, '
                    f'confidence: {confidence:.0%}.'
                )
            }

    def _identified_notification(self, context, fusion_result, time_str, time_of_day):
        """Generate notification for single-modality identification."""
        identity = context['identity']
        confidence = context['confidence']
        modality = fusion_result.get('modality', 'unknown')
        visit_count = context.get('visit_count', 0)
        is_returning = context.get('is_returning', False)
        greeting = self._get_greeting(time_of_day)

        modality_text = {
            'face_only': 'face recognition only',
            'voice_only': 'voice recognition only'
        }.get(modality, modality)

        if is_returning:
            return {
                'level': 'INFO',
                'icon': '\U0001f44b',  # 👋
                'message': f'{identity} detected again at {time_str} (via {modality_text})',
                'detail': (
                    f'{identity} identified via {modality_text} with {confidence:.0%} confidence. '
                    f'Visit #{visit_count}. Single-modality identification - '
                    f'consider verifying with additional modality.'
                )
            }
        elif context.get('is_new', False):
            return {
                'level': 'INFO',
                'icon': '\U0001f50d',  # 🔍
                'message': f'{identity} detected at {time_str} via {modality_text}',
                'detail': (
                    f'{greeting} {identity}! Identified via {modality_text} '
                    f'with {confidence:.0%} confidence. Note: single-modality '
                    f'identification - additional verification recommended.'
                )
            }
        else:
            return {
                'level': 'INFO',
                'icon': '\U0001f464',  # 👤
                'message': f'{identity} is present - {time_str}',
                'detail': (
                    f'Ongoing detection via {modality_text}. '
                    f'Confidence: {confidence:.0%}.'
                )
            }

    def _unknown_notification(self, context, fusion_result, time_str):
        """Generate notification for unrecognized person."""
        confidence = context['confidence']
        unknown_count = context.get('session_unknown_count', 0)

        if confidence > 0.3:
            return {
                'level': 'WARNING',
                'icon': '\u2753',  # ❓
                'message': f'Partially recognized person detected at {time_str}',
                'detail': (
                    f'A face was detected with {confidence:.0%} similarity to known persons, '
                    f'but below the recognition threshold. This could be a known person '
                    f'at an unusual angle or lighting. Consider enrolling if this is a new person.'
                )
            }
        else:
            return {
                'level': 'ALERT',
                'icon': '\U0001f195',  # 🆕
                'message': f'Unknown person detected at {time_str}',
                'detail': (
                    f'An unrecognized individual was detected with low confidence ({confidence:.0%}). '
                    f'Total unknown encounters this session: {unknown_count}. '
                    f'Consider enrolling this person or verifying their identity.'
                )
            }

    # ──────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────

    def _get_greeting(self, time_of_day):
        """Return a time-appropriate greeting."""
        greetings = {
            'morning': 'Good morning,',
            'afternoon': 'Good afternoon,',
            'evening': 'Good evening,',
            'night': 'Hello,'
        }
        return greetings.get(time_of_day, 'Hello,')

    def get_recent_notifications(self, n=10):
        """Get the N most recent notifications."""
        return self.history[-n:]

    def get_notification_counts(self):
        """Get counts of notifications by severity level."""
        counts = {'INFO': 0, 'WARNING': 0, 'ALERT': 0, 'CRITICAL': 0}
        for n in self.history:
            level = n.get('level', 'INFO')
            counts[level] = counts.get(level, 0) + 1
        return counts
