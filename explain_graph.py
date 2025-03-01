import csv
import json
import time
import os

import torch
import torch.nn as nn
from tqdm import tqdm
from datasets.load_datasets import get_dataset, get_dataloader
import torch_geometric.transforms as T
from models import GCN
from explainer import Explainer
from pgmexplainer import PGMExplainer
from pgexplainer import PGExplainer
from train_graph import evaluate

########################
import networkx as nx
import matplotlib.pyplot as plt


def visualize_graph(data, node_labels=None, figsize=(8, 8), node_size=500, font_size=12):
    """
    Visualizes a torch_geometric.data.Data object as a graph image.

    Args:
        data (torch_geometric.data.Data): The graph data object.
        node_labels (list or torch.Tensor, optional): Labels for the nodes. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (8, 8).
        node_size (int, optional): Size of the nodes. Defaults to 500.
        font_size (int, optional): Font size for node labels. Defaults to 12.
    """
    G = nx.Graph()

    # Add nodes
    num_nodes = data.num_nodes
    G.add_nodes_from(range(num_nodes))

    # Add edges
    edge_list = data.edge_index.t().tolist()  # Convert edge_index to list of tuples
    G.add_edges_from(edge_list)

    # Prepare node labels if provided
    labels = {}
    if node_labels is not None:
        if isinstance(node_labels, torch.Tensor):
            node_labels = node_labels.tolist()
        for i, label in enumerate(node_labels):
            labels[i] = label

    # Draw the graph
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G)  # Layout algorithm for node positioning
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=node_size, font_size=font_size)
    plt.show()

#######################



with open("configs.json") as config_file:
    configs = json.load(config_file)
    explainer_args = configs.get("explainer")
    dataset_name = configs.get("dataset_name").get("graph")

epochs = 5000
loop = True
pooling = {'mutagenicity': ['max', 'mean', 'sum'],
           'ba_2motifs': ['max'],
           'bbbp': ['max', 'mean', 'sum']}
if dataset_name == 'ba_2motifs':
    early_stop = 500
    loop = False
dataset = get_dataset(dataset_dir="./datasets", dataset_name=dataset_name)
dataset.data.x = dataset.data.x.float()
normalize = T.NormalizeFeatures()
dataset.data = normalize(dataset.data)
dataset.data.y = dataset.data.y.squeeze().long()
mode = explainer_args.get("mode")
node = bool(explainer_args.get("node"))
data_loader = get_dataloader(dataset, batch_size=1, random_split_flag=True,
                             data_split_ratio=[0.8, 0.1, 0.1], seed=2)

for dataset in data_loader['test']:
  visualize_graph(dataset)
  break

model = GCN(n_feat=dataset.num_node_features,
            n_hidden=20,
            n_class=dataset.num_classes,
            pooling=pooling[dataset_name.lower()],
            loop=loop)
loss_fc = nn.CrossEntropyLoss()
model_file = './src/' + dataset_name + '.pt'
model.load(model_file)
model.eval()

if mode == 0:
    explainer = Explainer(model, agg1=explainer_args.get("agg1"), agg2=explainer_args.get("agg2"),
                          lr=explainer_args.get("lr"), epochs=explainer_args.get("epochs"))
elif mode == 1:
    explainer = PGMExplainer(model, perturb_x_list=list(range(dataset.num_node_features)),
                             perturb_mode="zero")
else:
    duration = 0.
    explainer1 = PGExplainer(model, in_channels=40)
    tic = time.perf_counter()
    explainer1.train_explanation_network(data_loader['train'], label=0)
    duration += time.perf_counter() - tic
    explainer2 = PGExplainer(model, in_channels=40)
    tic = time.perf_counter()
    explainer2.train_explanation_network(data_loader['train'], label=1)
    duration += time.perf_counter() - tic
    print("duration:", duration)

acc_test, acc_loss = evaluate(data_loader['test'], model, loss_fc)
print(acc_test)

duration = 0.
results_path = "./node_masks/" + dataset_name + "/"
if not os.path.exists(results_path):
    os.makedirs(results_path)
print(dataset, "mode: ", mode, "node: ", node)

for i, data in enumerate(tqdm(data_loader['test'])):
    logit = model(data).view(-1)
    prediction = torch.argmax(logit, -1)
    tic = time.perf_counter()
    if mode == 0:
        feat_mask, edge_mask, node_mask = explainer.explain_graph(data, loss_fc=None,
                                                                  node=node, synchronize=explainer_args.get("synchronize"))
        file_path = results_path + str(i) + "_" + str(node) + ".csv"
    elif mode == 1:
        _, node_mask, _ = explainer.explain_graph(data,
                                                  num_samples=explainer_args.get("sample"), top_node=explainer_args.get("node_rate"))
        file_path = results_path + str(i) + "_" + "pgm" + ".csv"
    else:
        explainer = explainer1 if prediction == 0 else explainer2
        node_mask, _ = explainer.explain(data)
        file_path = results_path + str(i) + "_" + "pg" + ".csv"
    duration += time.perf_counter() - tic
    with open(file_path, "w", newline='') as filehandle:
        cw = csv.writer(filehandle)
        for listitem in node_mask.tolist():
            cw.writerow([listitem])
print("duration:", duration)
