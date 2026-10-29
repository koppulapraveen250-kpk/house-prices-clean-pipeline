"""BeatsPerMinute (AIS Competition v3) package initialization."""
from pathlib import Path

RAW_FEATURES: list[str] = [
    "RhythmScore",
    "AudioLoudness",
    "VocalContent",
    "AcousticQuality",
    "InstrumentalScore",
    "LivePerformanceLikelihood",
    "MoodScore",
    "TrackDurationMs",
    "Energy",
]

ENGINEERED_FEATURES: list[str] = [
    "Rhythm_Energy_Product",
    "Mood_Rhythm_Product",
    "Vocal_to_Instrumental_Ratio",
    "Acoustic_Energy_Ratio",
    "TrackDurationMinutes",
    "Loudness_Linear",
    "Energy_Mood_Sum",
]

FEATURE_COLUMNS: list[str] = RAW_FEATURES + ENGINEERED_FEATURES
ID_COLUMN: str = "id"
TARGET_COLUMN: str = "BeatsPerMinute"
MODELS_DIR: Path = Path(__file__).resolve().parent.parent / "models"
