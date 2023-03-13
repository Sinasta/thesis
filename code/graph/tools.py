import topologicpy
import topologic

from topologicpy.Topology import Topology
from topologicpy.Dictionary import Dictionary
from topologicpy.DGL import DGL
from topologicpy.Plotly import Plotly

from dgl import save_graphs, load_graphs
from dgl.data import CSVDataset
from tqdm.auto import tqdm
import torch
import json
import os
import warnings

warnings.filterwarnings("ignore")

def create_directories():
    if not os.path.exists('./graph/data/graph_batches'):
        os.makedirs('./graph/data/graph_batches')
        
    if not os.path.exists('./graph/data/geometry'):
        os.makedirs('./graph/data/geometry')
        
    if not os.path.exists('./graph/data/data_info'):
        os.makedirs('./graph/data/data_info')
        
    if not os.path.exists('./graph/data/graph_dataset'):
        os.makedirs('./graph/data/graph_dataset')
        
    if not os.path.exists('./graph/data/results'):
        os.makedirs('./graph/data/results')

def load_csv_dataset(validation_split=0.2):
    raw_dataset = CSVDataset('./graph/data/graph_dataset', force_reload=False)
    num_validation = int(len(raw_dataset) * validation_split)
    dgl_graphs = []
    graph_labels = []
    val_dgl_graphs = []
    val_graph_labels = []
    for number, element in enumerate(tqdm(raw_dataset, desc='rebuilding')):
        graph, label_dict = element
        label = label_dict['label'].tolist()
        
        graph.ndata['node_attr'] = graph.ndata['feat']
        graph.ndata.pop('feat')
        graph.ndata.pop('Y')
        graph.ndata.pop('X')
        
        if number < num_validation:
            val_dgl_graphs.append(graph)
            val_graph_labels.extend(label)
        else:
            dgl_graphs.append(graph)
            graph_labels.extend(label)
    dataset = DGL.DatasetByDGLGraphs(dgl_graphs, graph_labels, key='node_attr')
    validation_set = DGL.DatasetByDGLGraphs(val_dgl_graphs, val_graph_labels, key='node_attr')
    return dataset, validation_set

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
    
def build_dataset(amount, validation_split=0.2):
    num_validation = int(amount * validation_split)
    dgl_graphs = []
    graph_labels = []
    val_dgl_graphs = []
    val_graph_labels = []
    for number in range(amount):
        graph_list, label_dict = load_graphs('./graph/data/graph_batches/graph_batch_' + str(number) + '.bin')
        labels = label_dict['label'].tolist()
        if number < num_validation:
            val_dgl_graphs.extend(graph_list)
            val_graph_labels.extend(labels)
        else:
            dgl_graphs.extend(graph_list)
            graph_labels.extend(labels)
    dataset = DGL.DatasetByDGLGraphs(dgl_graphs, graph_labels, key='node_attr')
    validation_set = DGL.DatasetByDGLGraphs(val_dgl_graphs, val_graph_labels, key='node_attr')
    save_graphs('./graph/data/graph_dataset/graph_dataset.bin', dgl_graphs, {'label': torch.tensor(graph_labels)})
    save_graphs('./graph/data/graph_dataset/validation_dataset.bin', val_dgl_graphs, {'label': torch.tensor(val_graph_labels)})
    return dataset, validation_set
    
def balance_dataset(unbalanced_dataset):
    return DGL.BalanceDataset(unbalanced_dataset, unbalanced_dataset.labels.tolist(), method="undersampling")
    
def train(
    dataset,
    validationDataset=None,
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
    lr_decay=0.0,
    amsgrad=False):
    hyper_param = DGL.Hyperparameters(
        DGL.Optimizer(name=optimizer, amsgrad=amsgrad, betas=(0.9, 0.999), eps=1e-08, lr=lr, maximize=False, weightDecay=0.0, rho=0.9, lr_decay=lr_decay),
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
        './graph/data/results/classifier.pt', 
        './graph/data/results/results.csv')
    return DGL.Train(hyper_param, dataset, validationDataset)
    
def save_figs(results, dataset, validation_set, format_type='png', export=True):
    loss_labels = ['Epochs', 'Training Loss', 'Testing Loss']
    loss_df = Plotly.DataByDGL(data=results, labels=loss_labels)
    loss_fig = Plotly.FigureByDataFrame(loss_df, labels=loss_labels, title='Loss', x_title='Epochs', x_spacing=10.0, y_spacing=0.1, y_title='Loss')
    Plotly.Show(loss_fig, renderer='browser')
    acc_labels = ['Epochs', 'Training Accuracy', 'Testing Accuracy']
    acc_df = Plotly.DataByDGL(data=results, labels=acc_labels)
    acc_fig = Plotly.FigureByDataFrame(acc_df, labels=acc_labels, title='Accuracy', x_title='Epochs', x_spacing=10.0, y_spacing=0.1, y_title='Accuracy')
    Plotly.Show(acc_fig, renderer='browser')
    classifier = DGL.ClassifierByFilePath('./graph/data/results/classifier.pt')
    predicted = DGL.Predict(validation_set, classifier)['labels']
    actual = DGL.Labels(validation_set)
    accuracy = DGL.Accuracy(actual, predicted)
    cm = DGL.ConfusionMatrix(actual, predicted, normalize=True)
    cm_fig = Plotly.FigureByConfusionMatrix(cm, list(range(validation_set.gclasses)), showScale=False, title='Accuracy: ' + str(accuracy['accuracy']), colorScale='Viridis')
    Plotly.Show(cm_fig, renderer='browser')
    if export:
        cl = 'cl_' + str(len(dist['categories'][0]))
        ac = 'ac_' + str(accuracy['accuracy'])
        ep = 'ep_' + str(results['Epochs'][-1])
        ba = 'ba_' + str(results['Batch Size'][0])
        lr = 'lr_' + str(results['Learning Rate'][0])
        sz = 'sz_' + str(len(DGL.Labels(dataset)))
        folder_name = './graph/data/results/' + cl + '/' + ac + '_' + ep + '_' + ba + '_' + lr + '_' + sz
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        os.rename('./graph/data/results/classifier.pt', os.path.join(folder_name + '/' + 'classifier.pt'))
        os.rename('./graph/data/results/results.csv', os.path.join(folder_name + '/' + 'results.csv'))
        
        Plotly.ExportToImage(loss_fig, folder_name + '/loss.' + format_type, format=format_type, width='1280', height='720')
        Plotly.ExportToImage(acc_fig, folder_name + '/accuracy.'  + format_type, format=format_type, width='1280', height='720')
        Plotly.ExportToImage(cm_fig, folder_name + '/confusion_matrix.' + format_type, format=format_type, width='1280', height='720')
