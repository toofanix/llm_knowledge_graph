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

def df2ConceptList(dataframe: pd.DataFrame) -> List:
    results = dataframe.apply(
        lambda row: extractConcepts(
            row.text, {"chunk_id": row.chunk_id, "type":"concept"}
        ),
        axis=1
    )
    results = results.dropna()
    results = results.reset_index(drop=True)

    # Flatten the list of lists to single list
    concept_list = np.concatenate(results).ravel().tolist()
    return concept_list
