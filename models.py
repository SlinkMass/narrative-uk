from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
import numpy as np

class Article(BaseModel):
    id: str
    source: str
    headline: str
    summary: str
    url: str
    published_at: datetime
    bias_score: float
    embedding: Optional[np.ndarray] = None
    lexical_richness: float = 0.0  # 0.0 to 1.0
    readability_score: float = 0.0 # 0 to 100
    loaded_language_density: float = 0.0 # Percentage of text
    primary_signal: str = "Neutral"

    model_config = {
        "arbitrary_types_allowed": True  # allow np.ndarray
    }

class Story(BaseModel):
    story_id: str
    topic: str
    articles: List[Article]