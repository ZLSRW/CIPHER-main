from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from cipher import CIPHER, CIPHERConfig
from cipher.constants import SOURCE_NAMES, MODULE_COLUMNS
from cipher.data import CIPHERDataset
from cipher.losses import LossConfig, cipher_objective
from cipher.predict import load_checkpoint, predict_pairs
from cipher.train import train_fold


def small_config():
    return CIPHERConfig(node_feature_dim=8, field_dim=16, token_dim=8,
                        chebyshev_order=2, num_local_layers=1, num_operator_steps=2, dropout=0.0)


def small_graph():
    torch.manual_seed(9)
    n = 12
    left, right = torch.arange(n), torch.roll(torch.arange(n), -1)
    edges = torch.stack([torch.cat([left, right]), torch.cat([right, left])])
    adjacency = torch.zeros(n, n)
    adjacency[edges[0], edges[1]] = 1
    memberships = torch.randint(0, 2, (n, 6)).float()
    return Data(node_features=torch.randn(n, 8), topology_features=torch.rand(n, 4),
        asd_prior=torch.rand(n, 1), pathway_memberships=memberships,
        module_count=memberships.sum(dim=1, keepdim=True), source_coverage=torch.rand(n, 4),
        union_edge_index=edges, graph_laplacian=(torch.eye(n)-adjacency/2).to_sparse(),
        source_edge_indices={source:edges.clone() for source in SOURCE_NAMES}, num_nodes=n)


def test_pair_symmetry_contributions_and_gradients():
    graph = small_graph()
    model = CIPHER(small_config()).eval()
    pairs = torch.tensor([[0, 3], [2, 6], [4, 8], [1, 9]])
    output = model(graph, pairs)
    torch.testing.assert_close(output["pair_logits"], model(graph, pairs.flip(1))["pair_logits"])
    decoded = model.relational_decoder
    reconstructed = (decoded.pair_bias + output["source_factor_contributions"].mean(dim=(1, 2))
                     + decoded.g_mech(output["pair_representation"]).squeeze(-1))
    torch.testing.assert_close(reconstructed, output["pair_logits"])
    assert output["source_factor_contributions"].shape == (4, 4, 7)
    assert output["pair_representation"].shape == (4, 40)
    loss = cipher_objective(output, torch.tensor([1., 0., 1., 0.]), 6)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.structured_operator.steps[0].phi_d[0].weight.grad.abs().sum() > 0
    assert model.structured_operator.steps[0].csst.g_rho[0].weight.grad.abs().sum() > 0
    cached = predict_pairs(model, graph, pairs, batch_size=2)
    np.testing.assert_allclose(cached["pair_logits"], output["pair_logits"].detach().numpy(), atol=1e-6)


def test_config_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        CIPHERConfig(token_dim=7)
    with pytest.raises(ValueError):
        CIPHERConfig(num_pathway_factors=5)


def write_tiny_dataset(directory):
    graph = small_graph()
    n = graph.num_nodes
    pd.DataFrame({"node_id":range(n), "symbol":[f"PROT{i}" for i in range(n)],
                  "feature_8":[",".join(map(str,row)) for row in graph.node_features.tolist()]}).to_csv(directory/"PPI_node.csv",index=False)
    edge_pairs = graph.union_edge_index[:, :n].T.numpy()
    edges = pd.DataFrame(edge_pairs,columns=["left","right"])
    edges.to_csv(directory/"PPI_edge.csv",index=False)
    for source in SOURCE_NAMES:
        edges[source] = 1
    edges.to_csv(directory/"edge_source_flags.csv",index=False)
    topology = pd.DataFrame(graph.topology_features.numpy(),columns=["log_degree","pagerank","clustering","core_number"])
    topology.insert(0,"node_id",range(n)); topology.to_csv(directory/"topology_features.csv",index=False)
    pd.DataFrame({"node_id":range(n),"sfari_prior":graph.asd_prior[:,0].numpy()}).to_csv(directory/"sfari_node_prior.csv",index=False)
    modules = pd.DataFrame(graph.pathway_memberships.numpy(),columns=MODULE_COLUMNS)
    modules.insert(0,"node_id",range(n)); modules.to_csv(directory/"reactome_module_matrix.csv",index=False)
    coverage = pd.DataFrame(graph.source_coverage.numpy(),columns=["string_cov","biogrid_cov","intact_cov","consensus_cov"])
    coverage.insert(0,"node_id",range(n)); coverage.to_csv(directory/"node_source_coverage.csv",index=False)
    (directory/"metadata.json").write_text('{"dataset_name":"synthetic_software_test"}')
    split = pd.DataFrame([[0,3,1],[1,4,0],[2,5,1],[3,6,0],[4,7,1],[5,8,0],[6,9,1],[7,10,0]],columns=["left","right","label"])
    fold = directory/"splits/fold_1"; fold.mkdir(parents=True)
    split.to_csv(fold/"train.csv",index=False)
    split.iloc[:4].assign(left=[0,1,2,3],right=[5,6,7,8]).to_csv(fold/"val.csv",index=False)
    split.iloc[:4].assign(left=[0,1,2,3],right=[6,7,8,9]).to_csv(directory/"split_test.csv",index=False)


def test_train_save_load_predict_pipeline(tmp_path):
    write_tiny_dataset(tmp_path)
    configuration = {"model":asdict(small_config()), "loss":asdict(LossConfig()),
        "training":{"epochs":2,"patience":2,"learning_rate":0.0007,"weight_decay":0.0001,"gradient_clip":2.0,"seed":42}}
    metrics = train_fold(tmp_path,1,configuration,tmp_path/"output","cpu")
    assert 0 <= metrics["auc"] <= 1
    model, checkpoint = load_checkpoint(tmp_path/"output/cipher.pt",torch.device("cpu"))
    dataset = CIPHERDataset(tmp_path,fold=None)
    pairs = torch.tensor([[0,6],[1,7]])
    result = predict_pairs(model,dataset.data,pairs,batch_size=1)
    assert result["probability"].shape == (2,)
    assert checkpoint["seed"] == 42
    assert np.isfinite(result["pair_representation"]).all()
