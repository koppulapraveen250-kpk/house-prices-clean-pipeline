"""House prices package initialization."""
from pathlib import Path

CONTINUOUS_FEATURES: list[str] = ["GrLivArea", "LotArea"]
CATEGORICAL_FEATURES: list[str] = ["Neighborhood", "HouseStyle"]
FEATURE_COLUMNS: list[str] = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
LABEL_COLUMN: str = "SalePrice"
MODELS_DIR: Path = Path(__file__).resolve().parent.parent / "models"
