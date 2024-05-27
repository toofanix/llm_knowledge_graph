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

def df2Graph(dataframe: pd.DataFrame, model=None)->list:
    results = dataframe.apply(
        lambda row: graphPrompt(row.text, {"chunk_id":row.chunk_id}, model), axis=1
    )

    results = results.dropna()
    results = results.reset_index(drop=True)

    # Flattent the list of lists to a single list
    concept_list = np.concatenate(results).ravel().tolist()
    return concept_list

def graph2Df(nodes_list) -> pd.DataFrame:
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
    graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(lambda x: x.lower())
    graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(lambda x: x.lower())
    return graph_dataframe