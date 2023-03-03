import topologicpy
import topologic

from topologicpy.Topology import Topology
from topologicpy.Dictionary import Dictionary
from topologicpy.DGL import DGL
import torch
from dgl import save_graphs, load_graphs

import json
import os

if not os.path.exists('./graph/data/graph_batches'):
    os.makedirs('./graph/data/graph_batches')
    
if not os.path.exists('./graph/data/graph_dataset'):
    os.makedirs('./graph/data/graph_dataset')

def save_graph_batch(ccomplex_database, name):
    dgl_graphs = []
    labels = []
    for ccomplex_variant_list in ccomplex_database.values():
            for ccomplex_tuple in ccomplex_variant_list:
                ccomplex, graph, energy_class = ccomplex_tuple
                dgl_graph = DGL.ByGraph(graph, bidirectional=True, key='label', categories=categories(), node_attr_key='node_attr', tolerance=0.0001)
                dgl_graphs.append(dgl_graph)
                labels.append(energy_class)
    graph_path = './graph/data/graph_batches/graph_batch_' + name + '.bin'
    graph_labels = {'label': torch.tensor(labels)}
    save_graphs(graph_path, dgl_graphs, graph_labels)

def categories():
    with open('./graph/conversion_list.json', 'r') as cl:
        conversion_list = json.load(cl)
    return list(range(len(conversion_list)))
    
def build_dataset(amount):
    dgl_graphs = []
    graph_labels = []
    for filename in os.listdir('./graph/data/graph_batches/'):
        number = int(filename.split('_')[2].split('.')[0])
        if number in list(range(amount)):
            graph_list, label_dict = load_graphs('./graph/data/graph_batches/' + filename)
            labels = label_dict['label'].tolist()
            dgl_graphs.extend(graph_list)
            graph_labels.extend(labels)
    dataset = DGL.DatasetByDGLGraphs(dgl_graphs, graph_labels, key='node_attr')
    save_graphs('./graph/data/graph_dataset/graph_dataset.bin', dgl_graphs, {'label': torch.tensor(graph_labels)})
    return dataset
    
def train(dataset, batch_size=50, epochs=100, lr=0.01, cv_type='Holdout', split=0.2, k_folds=5, hl_widths=[32], conv_layer_type='GraphConv', pooling='AvgPooling', use_gpu=False, loss_function='Cross Entropy'):
    hyper_param = DGL.Hyperparameters(
        DGL.Optimizer(lr=0.01),
        cv_type,
        split,
        k_folds,
        hl_widths,
        conv_layer_type,
        pooling,
        batch_size, 
        epochs,
        use_gpu,
        loss_function,
        './graph/data/classifier/classifier.pt', 
        './graph/data/results/results.csv')
    return DGL.Train(hyper_param, dataset)
