import uuid
import pandas as pd
import numpy as np
from prompts import extractConcepts, graphPrompt

def documents2Dataframe(documents) -> pd.Dataframe:
    rows = []
    for chunk in documents:
        row = {
            "text": chunk.page_content,
            **chunk.metadata,
            "chunk_id": uuid.uuid4().hex
        }
        rows = rows + row
    df = pd.DataFrame(rows)
    return df

