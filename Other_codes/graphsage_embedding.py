import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
import joblib

DATASETS = {
    "cresci17": {
        "node_file": "cresci17_node_features.csv",
        "edge_file": "cresci17_similarity_edges.csv",
        "label_file": "cresci17_labels.csv",
        "emb_out": "cresci17_node_embeddings.csv"
    },
    "twibot22": {
        "node_file": "twibot22_node_features.csv",
        "edge_file": "twibot22_similarity_edges.csv",
        "label_file": "twibot22_labels.csv",
        "emb_out": "twibot22_node_embeddings.csv"
    }
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
LR = 0.005
HIDDEN_DIM = 128
EMB_DIM = 64

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, emb_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, emb_dim)
        self.classifier = torch.nn.Linear(emb_dim, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        z = self.conv2(x, edge_index)
        return z

    def predict(self, z):
        return self.classifier(z)


def load_graph(node_file, edge_file, label_file):

    nodes = pd.read_csv(node_file)
    edges = pd.read_csv(edge_file)
    labels_df = pd.read_csv(label_file)

    nodes["user_id"] = nodes["user_id"].astype(str)
    edges["src"] = edges["src"].astype(str)
    edges["dst"] = edges["dst"].astype(str)
    labels_df["user_id"] = labels_df["user_id"].astype(str)

    node_ids = nodes["user_id"].values
    id_map = {uid: i for i, uid in enumerate(node_ids)}

    scaler = StandardScaler()
    x = torch.tensor(
        scaler.fit_transform(nodes.drop(columns=["user_id"]).values),
        dtype=torch.float
    )
    joblib.dump(scaler, "graph_scaler.pkl")

    labels_df = labels_df.set_index("user_id")
    y_series = labels_df.reindex(node_ids)["label"]
    y = torch.tensor(y_series.values, dtype=torch.long)

    train_idx, test_idx = train_test_split(
        np.arange(len(node_ids)),
        test_size=0.2,
        stratify=y.numpy(),
        random_state=42
    )

    train_ids = node_ids[train_idx]
    test_ids = node_ids[test_idx]

    pd.DataFrame({"user_id": train_ids}).to_csv(f"{node_file}_train_ids.csv", index=False)
    pd.DataFrame({"user_id": test_ids}).to_csv(f"{node_file}_test_ids.csv", index=False)

    train_mask = torch.zeros(len(node_ids), dtype=torch.bool)
    test_mask = torch.zeros(len(node_ids), dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    valid_edges = edges[
        edges["src"].isin(id_map) & edges["dst"].isin(id_map)
    ]

    edge_index = torch.tensor(
        [[id_map[s], id_map[d]] for s, d in zip(valid_edges["src"], valid_edges["dst"])],
        dtype=torch.long
    ).t().contiguous()

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask
    )

    return data, node_ids

if __name__ == "__main__":
    graphs = {}
    input_dim = None

    for name, cfg in DATASETS.items():
        data, ids = load_graph(cfg["node_file"], cfg["edge_file"], cfg["label_file"])
        data = data.to(DEVICE)
        graphs[name] = (data, ids)

        if input_dim is None:
            input_dim = data.x.shape[1]

    model = GraphSAGE(input_dim, HIDDEN_DIM, EMB_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        total_loss = 0

        for data, _ in graphs.values():
            z = model(data.x, data.edge_index)
            logits = model.predict(z)
            loss = criterion(logits[data.train_mask], data.y[data.train_mask])
            total_loss += loss

        total_loss = total_loss / len(graphs)
        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} Loss {total_loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        for name, (data, ids) in graphs.items():
            z = model(data.x, data.edge_index).cpu().numpy()
            emb_df = pd.DataFrame(z)
            emb_df.insert(0, "user_id", ids)
            emb_df.to_csv(DATASETS[name]["emb_out"], index=False)

    print("GraphSAGE embeddings saved.")
    torch.save(model.state_dict(), "graphsage_model.pt")
