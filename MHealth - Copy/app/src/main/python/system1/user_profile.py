"""
User Profile — Self-report data from Lumen app onboarding.

Collected during onboarding and periodic check-ins. Used to:
  - Flag baseline contamination (PHQ-9/GAD-7 at onboarding)
  - Adjust feature weights (lifestyle scores)
  - Context-aware filtering (exam periods, life events)
  - Demographic-adjusted norms

This data modulates pipeline behavior — it does NOT replace sensor detection.
The 29-feature PersonalityVector stays pure (sensor-only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class UserProfile:
    """Self-report data collected during Lumen app onboarding."""

    # ── Clinical screening ──────────────────────────────────────────────
    phq9_score: int = 0                # 0-27
    gad7_score: int = 0                # 0-21
    phq9_item_scores: List[int] = field(default_factory=lambda: [0] * 9)
    gad7_item_scores: List[int] = field(default_factory=lambda: [0] * 7)

    # ── Demographics ───────────────────────────────────────────────────
    age: int = 25
    gender: str = 'prefer_not_to_say'
    living_situation: str = 'with_family'   # alone/with_family/roommates/hostel
    employment: str = 'student'             # student/full_time/part_time/self_employed/unemployed

    # ── Routine ────────────────────────────────────────────────────────
    typical_wake: float = 7.0           # hour
    typical_sleep: float = 23.0         # hour
    commute_minutes: int = 30
    routine_consistency: str = 'flexible'  # rigid/flexible/variable

    # ── Lifestyle weights (0-5 scale) ─────────────────────────────────
    lifestyle_screen: int = 3           # phone reliance
    lifestyle_communication: int = 3    # call/msg activity
    lifestyle_movement: int = 3         # physical movement
    lifestyle_sleep: int = 3            # sleep hygiene
    lifestyle_behavioral: int = 3       # digital-mood reflection
    lifestyle_engagement: int = 3       # wellness check-in likelihood

    # ── Contextual anchors ────────────────────────────────────────────
    is_student: bool = False
    exam_period: Optional[Tuple[str, str]] = None  # (start_date, end_date) ISO format
    recent_life_event: bool = False
    has_chronic_condition: bool = False
    in_therapy: bool = False
    physical_health_rating: int = 7     # 1-10

    # ── Computed properties ───────────────────────────────────────────

    @property
    def is_baseline_contaminated(self) -> bool:
        """True if PHQ-9 or GAD-7 indicate clinical-level symptoms."""
        return self.phq9_score >= 10 or self.gad7_score >= 10

    @property
    def contamination_type(self) -> Optional[str]:
        """Returns likely contamination type based on screening scores."""
        if self.phq9_score >= 15:
            return 'depression'
        if self.gad7_score >= 15:
            return 'anxiety'
        if self.phq9_score >= 10:
            return 'depression_mild'
        if self.gad7_score >= 10:
            return 'anxiety_mild'
        return None

    @property
    def has_suicidal_ideation(self) -> bool:
        """PHQ-9 item 9 (thoughts of death/self-harm) scored > 0."""
        return len(self.phq9_item_scores) >= 9 and self.phq9_item_scores[8] > 0

    def get_lifestyle_dict(self) -> Dict[str, int]:
        """Returns lifestyle scores as a dict keyed by category."""
        return {
            'screen': self.lifestyle_screen,
            'communication': self.lifestyle_communication,
            'movement': self.lifestyle_movement,
            'sleep': self.lifestyle_sleep,
            'behavioral': self.lifestyle_behavioral,
            'engagement': self.lifestyle_engagement,
        }

    def is_exam_period_active(self, current_date: Optional[str] = None) -> bool:
        """Check if current date falls within declared exam period."""
        if self.exam_period is None:
            return False
        from datetime import datetime
        try:
            start = datetime.fromisoformat(self.exam_period[0])
            end = datetime.fromisoformat(self.exam_period[1])
            current = datetime.fromisoformat(current_date) if current_date else datetime.now()
            return start <= current <= end
        except (ValueError, IndexError):
            return False
