# experiments/run_ogb_mag_linkteller_relmask.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.nn import HeteroConv, GCNConv

from src.config import DEVICE, DEFAULT_DELTA, DEFAULT_AGG, DEFAULT_USE_PROB
from src.logingteller import linkteller_scores_for_hetero  # adjust import path
from src.metrics import per_relation_f1_report
from src.hetero_training import make_hetero_gbb_api_for_model


class SimpleHeteroGCN(nn.Module):
    def __init__(self, metadata, hidden_channels=64, out_channels=2, dropout=0.3):
        super().__init__()
        self.metadata = metadata
        self.dropout = dropout

        node_types, edge_types = metadata

        # First layer
        convs1 = {}
        for edge_type in edge_types:
            convs1[edge_type] = GCNConv(-1, hidden_channels)
        self.conv1 = HeteroConv(convs1, aggr="sum")

        # Second layer
        convs2 = {}
        for edge_type in edge_types:
            convs2[edge_type] = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = HeteroConv(convs2, aggr="sum")

        # Linear heads per node type
        self.lins = nn.ModuleDict()
        for ntype in node_types:
            self.lins[ntype] = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(x) for k, x in x_dict.items()}
        x_dict = {k: F.dropout(x, p=self.dropout, training=self.training)
                  for k, x in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.relu(x) for k, x in x_dict.items()}

        out_dict = {}
        for ntype, x in x_dict.items():
            out_dict[ntype] = self.lins[ntype](x)
        return out_dict


def load_ogb_mag(root="data/OGB"):
    dataset = PygNodePropPredDataset(name="ogbn-mag", root=root)
    data = dataset[0]
    return data, dataset.num_classes


def train_paper_classifier(model, data, epochs=50, lr=1e-2, weight_decay=5e-4):
    model = model.to(DEVICE)
    x_dict = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}

    paper_y = data["paper"].y.view(-1)
    train_mask = data["paper"].train_mask

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits_dict = model(x_dict, edge_index_dict)
        logits_paper = logits_dict["paper"]
        loss = F.cross_entropy(
            logits_paper[train_mask.to(DEVICE)],
            paper_y[train_mask].to(DEVICE),
        )
        loss.backward()
        opt.step()

        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits_dict_eval = model(x_dict, edge_index_dict)
                logits_paper_eval = logits_dict_eval["paper"]
                preds = logits_paper_eval.argmax(dim=-1).cpu()
                train_acc = (
                    (preds[train_mask] == paper_y[train_mask]).float().mean().item()
                )
            print(f"[Ep {ep}] loss={loss.item():.4f} train_acc={train_acc:.4f}")

    return model


def build_paper_relation_edges(data):
    """
    Build relation_edge_index_dict for paper nodes only.

    We keep only edge types where src and dst are both "paper".
    These are usually ("paper", "cites", "paper") and possibly its reverse.
    """
    rel_ei = {}
    for edge_type, ei in data.edge_index_dict.items():
        src_nt, rel_name, dst_nt = edge_type
        if src_nt == "paper" and dst_nt == "paper":
            key = f"{src_nt}__{rel_name}__{dst_nt}"
            rel_ei[key] = ei
    return rel_ei


def main():
    data, num_classes = load_ogb_mag()
    metadata = (list(data.x_dict.keys()), list(data.edge_index_dict.keys()))

    model = SimpleHeteroGCN(metadata, hidden_channels=64, out_channels=2)
    model = train_paper_classifier(model, data, epochs=50)

    # Prepare features and GBB API
    x_dict = {k: v.clone() for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.clone() for k, v in data.edge_index_dict.items()}

    gbb_api = make_hetero_gbb_api_for_model(
        model=model,
        edge_index_dict=edge_index_dict,
        target_ntype="paper",
    )

    paper_x = x_dict["paper"]
    N_paper = paper_x.size(0)

    # Relation masking example: you can pass rel_mask=None to use all
    score_dict = linkteller_scores_for_hetero(
        gbb_api=gbb_api,
        x_dict=x_dict,
        target_ntype="paper",
        delta=DEFAULT_DELTA,
        vi_nodes=None,
        agg=DEFAULT_AGG,
        use_prob=DEFAULT_USE_PROB,
        rel_mask=None,
    )

    # Build per relation edge indices for paper-paper relations
    relation_edge_index_dict = build_paper_relation_edges(data)

    report = per_relation_f1_report(
        score_dict=score_dict,
        relation_edge_index_dict=relation_edge_index_dict,
        num_nodes=N_paper,
        top_k_mode="num_edges",
    )

    print("Per relation LinkTeller F1 on paper-paper edges:")
    for rel_name, stats in report.items():
        print(
            f"{rel_name}: "
            f"F1={stats['f1']:.4f} "
            f"Prec={stats['precision']:.4f} "
            f"Rec={stats['recall']:.4f} "
            f"|E|={int(stats['num_edges'])}"
        )

    # Example of relation masking run, per relation
    for edge_type in data.edge_index_dict.keys():
        if edge_type[0] == "paper" and edge_type[2] == "paper":
            rel_mask = [edge_type]
            score_rel = linkteller_scores_for_hetero(
                gbb_api=gbb_api,
                x_dict=x_dict,
                target_ntype="paper",
                delta=DEFAULT_DELTA,
                vi_nodes=None,
                agg=DEFAULT_AGG,
                use_prob=DEFAULT_USE_PROB,
                rel_mask=rel_mask,
            )
            rel_name = f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"
            report_rel = per_relation_f1_report(
                score_dict=score_rel,
                relation_edge_index_dict={rel_name: relation_edge_index_dict[rel_name]},
                num_nodes=N_paper,
                top_k_mode="num_edges",
            )
            stats = report_rel[rel_name]
            print(
                f"[MASK {rel_name} only] F1={stats['f1']:.4f} "
                f"Prec={stats['precision']:.4f} "
                f"Rec={stats['recall']:.4f}"
            )


if __name__ == "__main__":
    main()
