import pandas as pd
from sklearn.neighbors import NearestNeighbors

INPUT_FILE = r"D:\Final year project\code\cleaned_project_dataset.xlsx"
K = 10

DATASETS = {
    "cresci17": {
        "edge_out": "cresci17_similarity_edges.csv",
        "feat_out": "cresci17_node_features.csv",
        "label_out": "cresci17_labels.csv"
    },
    "twibot22": {
        "edge_out": "twibot22_similarity_edges.csv",
        "feat_out": "twibot22_node_features.csv",
        "label_out": "twibot22_labels.csv"
    }
}

df = pd.read_excel(INPUT_FILE)

NON_FEATURE_COLS = ["user_id", "user_name", "Label", "dataset"]
FEATURE_COLS = [c for c in df.columns if c not in NON_FEATURE_COLS]

for ds_name, paths in DATASETS.items():

    ds_df = df[df["dataset"] == ds_name].reset_index(drop=True)

    user_ids = ds_df["user_id"].astype(str).values
    labels = ds_df["Label"].values

    feature_df = ds_df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    X = feature_df.values

    knn = NearestNeighbors(n_neighbors=K + 1, metric="cosine")
    knn.fit(X)

    distances, indices = knn.kneighbors(X)

    edges = []

    for i in range(len(user_ids)):
        for j in range(1, K + 1):
            dst_idx = indices[i][j]
            weight = 1 - distances[i][j]
            edges.append([user_ids[i], user_ids[dst_idx], weight])
            edges.append([user_ids[dst_idx], user_ids[i], weight])

    pd.DataFrame(edges, columns=["src", "dst", "weight"]).to_csv(paths["edge_out"], index=False)

    feature_df.assign(user_id=user_ids).to_csv(paths["feat_out"], index=False)

    pd.DataFrame({"user_id": user_ids, "label": labels}).to_csv(paths["label_out"], index=False)

print("Similarity graphs built.")
