import topologicpy
import topologic

from topologicpy.Topology import Topology
from topologicpy.Dictionary import Dictionary
from topologicpy.DGL import DGL
from topologicpy.Plotly import Plotly

from dgl import save_graphs, load_graphs
import torch
import json
import os
import pandas as pd

if not os.path.exists('./graph/data/graph_batches'):
    os.makedirs('./graph/data/graph_batches')
    
if not os.path.exists('./graph/data/graph_dataset'):
    os.makedirs('./graph/data/graph_dataset')
    
if not os.path.exists('./graph/data/classifier'):
    os.makedirs('./graph/data/classifier')
    
if not os.path.exists('./graph/data/results'):
    os.makedirs('./graph/data/results')

def save_graph_batch(ccomplex_database, name):
    dgl_graphs = []
    labels = []
    for ccomplex_variant_list in ccomplex_database.values():
            for ccomplex_tuple in ccomplex_variant_list:
                ccomplex, graph, energy_class = ccomplex_tuple
                dgl_graph = DGL.ByGraph(graph, bidirectional=True, key='label', categories=list(range(91)), node_attr_key='node_attr', tolerance=0.0001)
                dgl_graphs.append(dgl_graph)
                labels.append(energy_class)
    graph_path = './graph/data/graph_batches/graph_batch_' + name + '.bin'
    graph_labels = {'label': torch.tensor(labels)}
    save_graphs(graph_path, dgl_graphs, graph_labels)
    
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
    
def balance_dataset(unbalanced_dataset):
    return DGL.BalanceDataset(unbalanced_dataset, unbalanced_dataset.labels.tolist(), method="undersampling")
    
def train(
    dataset,
    batch_size=50,
    epochs=100,
    lr=0.01,
    cv_type='Holdout',
    split=0.2,
    k_folds=5,
    hl_widths=[32],
    conv_layer_type='GraphConv',
    pooling='AvgPooling',
    use_gpu=False,
    loss_function='Cross Entropy',
    optimizer='Adam',
    lr_decay=0.0):
    hyper_param = DGL.Hyperparameters(
        DGL.Optimizer(name=optimizer, amsgrad=False, betas=(0.9, 0.999), eps=1e-08, lr=lr, maximize=False, weightDecay=0.0, rho=0.9, lr_decay=lr_decay),
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
    
def save_figs(results, dataset):
    loss_fig = DGL.Show(results, labels=['Epochs', 'Training Loss', 'Testing Loss'], title='Loss', x_title='Epoch', y_title='Loss', renderer='browser')
    Plotly.ExportToImage(loss_fig, './graph/data/results/loss.svg', format='svg', width='1280', height='720')
    acc_fig = DGL.Show(results, labels=['Epochs', 'Training Accuracy', 'Testing Accuracy'], title='Accuracy', x_title='Epoch', y_title='Accuracy', renderer='browser')
    Plotly.ExportToImage(acc_fig, './graph/data/results/accuracy.svg', format='svg', width='1280', height='720')
    labels = DGL.Labels(dataset)
    dist = DGL.CategoryDistribution(labels, categories=None, mantissa=4)
    dist_fig = Plotly.FigureByPieChart(dist['ratios'][0], dist['categories'][0])
    Plotly.Show(dist_fig, renderer='browser')
    Plotly.ExportToImage(dist_fig, './graph/data/results/distribution.svg', format='svg', width='1280', height='720')
    classifier = DGL.ClassifierByFilePath('./graph/data/classifier/classifier.pt')
    predicted = DGL.Predict(dataset, classifier)['labels']
    cm = DGL.ConfusionMatrix(labels, predicted, normalize=True)
    cm_fig = Plotly.FigureByConfusionMatrix(cm, list(range(2)), showScale=False)
    Plotly.Show(cm_fig, renderer='browser')
    Plotly.ExportToImage(cm_fig, './graph/data/results/confusion_matrix.svg', format='svg', width='1280', height='720')
    accuracy = DGL.Accuracy(labels, predicted)
    df = pd.DataFrame([accuracy['size'], accuracy['correct'], accuracy['wrong']], ['Size', 'Correct', 'Wrong'], ['Metric'])
    acc_info_fig = Plotly.FigureByDataFrame(df, labels=['Metric', 'Size', 'Correct', 'Wrong'], title='Accuracy: ' + str(accuracy['accuracy']), x_title='Amount', x_spacing=200.0, y_title='', chart_type='Bar')
    Plotly.Show(acc_info_fig, renderer='browser')
    Plotly.ExportToImage(acc_info_fig, './graph/data/results/accuracy_info.svg', format='svg', width='1280', height='720')
