import topologicpy
import topologic

from topologicpy.Topology import Topology
from topologicpy.Dictionary import Dictionary
from topologicpy.DGL import DGL
from topologicpy.Plotly import Plotly

import dgl
from dgl import save_graphs, load_graphs
from dgl.data import CSVDataset
from tqdm.auto import tqdm
import torch
import json
import os
import warnings
import pandas

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
    dataset = DGL.DatasetByGraphs({'graphs':dgl_graphs, 'labels': graph_labels}, key='node_attr')
    validation_set = DGL.DatasetByGraphs({'graphs':val_dgl_graphs, 'labels': val_graph_labels}, key='node_attr')
    return dataset, validation_set

def save_graph_batch(ccomplex_database, name, num_nlabels):
    dgl_graphs = []
    labels = []
    values = []
    for ccomplex_variant_list in ccomplex_database.values():
            for ccomplex_tuple in ccomplex_variant_list:
                ccomplex, graph, energy_class, consumption = ccomplex_tuple
                dgl_graph = DGL.GraphByTopologicGraph(graph, bidirectional=True, key='label', categories=list(range(num_nlabels)), node_attr_key='node_attr', tolerance=0.0001)
                dgl_graphs.append(dgl_graph)
                labels.append(energy_class)
                values.append(consumption)
    graph_path = './graph/data/graph_batches/graph_batch_' + name + '.bin'
    graph_labels = {'label': torch.tensor(labels), 'value': torch.tensor(values)}
    save_graphs(graph_path, dgl_graphs, graph_labels)
    
def build_dataset(amount, fracList=[0.8, 0.1, 0.1], shuffle=False, randomState=1, label_key='label'):
    dgl_graphs = []
    graph_labels = []
    for number in range(amount):
        graph_list, label_dict = load_graphs('./graph/data/graph_batches/graph_batch_' + str(number) + '.bin')
        labels = label_dict[label_key].tolist()
        dgl_graphs.extend(graph_list)
        graph_labels.extend(labels)  
    dataset = DGL.DatasetByGraphs({'graphs':dgl_graphs, 'labels': graph_labels}, key='node_attr')
    datasets = DGL.DatasetSplit(dataset, fracList=fracList, shuffle=shuffle, randomState=randomState, key="node_attr")
    save_graphs('./graph/data/graph_dataset/train_dataset.bin', datasets['train_ds'].graphs, {label_key: datasets['train_ds'].labels})
    save_graphs('./graph/data/graph_dataset/validate_dataset.bin', datasets['validate_ds'].graphs, {label_key: datasets['validate_ds'].labels})
    save_graphs('./graph/data/graph_dataset/test_dataset.bin', datasets['test_ds'].graphs, {label_key: datasets['test_ds'].labels})
    return datasets
    
def balance_dataset(unbalanced_dataset):
    return DGL.BalanceDataset(unbalanced_dataset, unbalanced_dataset.labels.tolist(), method="undersampling")

def train(datasets, batch_size=50, epochs=100, lr=0.01, cv_type='Holdout', k_folds=10, hl_widths=[32], conv_layer_type='GraphConv', pooling='AvgPooling', use_gpu=False, loss_function='Cross Entropy', model_type='Classifier'):
    optimizer = DGL.Optimizer(name='Adam', amsgrad=False, betas=(0.9, 0.999), eps=1e-08, lr=lr, maximize=False, weightDecay=0.0, rho=0.9, lr_decay=0.0)
    hyper_param = DGL.Hyperparameters(optimizer, model_type, cv_type, [0.8,0.1,0.1], k_folds, hl_widths, conv_layer_type, pooling, batch_size, epochs, use_gpu, loss_function)
    model = DGL.Model(hyper_param, datasets['train_ds'], datasets['validate_ds'], datasets['test_ds'])
    model = DGL.ModelTrain(model)
    model = DGL.ModelTest(model)
    DGL.ModelSave(model, path='./graph/data/results/' + model_type.lower() + '.pt')
    results = DGL.ModelData(model)
    dataloaders = None
    if cv_type.lower() == 'k-fold':
        dataloaders = model.validate_dataloaders
    return results, dataloaders
    
def save_figs(results, dataloaders, datasets, format_type='png', export=True, task='classification', cv_type='K-Fold'):
    if task.lower() == 'classification':
        acc_labels = ['Epochs', 'Training Accuracy', 'Validation Accuracy', 'Testing Accuracy']
        acc_df = Plotly.DataByDGL(data=results, labels=acc_labels)
        acc_fig = Plotly.FigureByDataFrame(acc_df, labels=acc_labels, title='Accuracy', xTitle='Epochs', xSpacing=10.0, ySpacing=0.1, yTitle='Accuracy')
        Plotly.Show(acc_fig, renderer='browser')
        loss_labels = ['Epochs', 'Training Loss', 'Validation Loss', 'Testing Loss']
        loss_df = Plotly.DataByDGL(data=results, labels=loss_labels)
        loss_fig = Plotly.FigureByDataFrame(loss_df, labels=loss_labels, title='Loss', xTitle='Epochs', xSpacing=10.0, ySpacing=0.1, yTitle='Loss')
        Plotly.Show(loss_fig, renderer='browser')
        classifier = DGL.ModelByFilePath('./graph/data/results/classifier.pt')
        predicted = DGL.ModelClassify(classifier, datasets['test_ds'])['predictions']
        actual = DGL.DatasetLabels(datasets['test_ds'])
        act_pred_df = pandas.DataFrame({'index' : list(range(len(actual))), 'predicted' : predicted, 'actual' : actual})
        cm = DGL.ConfusionMatrix(actual, predicted, normalize=True)
        cm_fig = Plotly.FigureByConfusionMatrix(cm, list(range(datasets['test_ds'].gclasses)), showScale=False, title='Confusion Matrix', colorScale='Viridis')
        Plotly.Show(cm_fig, renderer='browser')
        if export:
            cl = 'classification'
            ac = 'ac_' + str(results['Testing Accuracy'][0][0]).replace('.', '_')
            ep = 'ep_' + str(results['Epochs'][-1])
            ba = 'ba_' + str(results['Batch Size'][0])
            lr = 'lr_' + str(results['Learning Rate'][0])
            sz = 'sz_' + str(len(DGL.DatasetLabels(datasets['train_ds'])))
            folder_name = './graph/data/results/' + cl + '/' + ac + '_' + ep + '_' + ba + '_' + lr + '_' + sz
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            
            if cv_type.lower() == 'k-fold':
                for fold, dataloader in enumerate(dataloaders):
                    classifier = DGL.ModelByFilePath('./graph/data/results/classifier_' + str(fold) + '.pt')
                    actual_list = []
                    predicted_list = []
                    for batched_graph, labels in dataloader:
                        actual_list.extend(labels.tolist())
                        pred = classifier(batched_graph, batched_graph.ndata['node_attr'].float())
                        out, classes = torch.max(pred, dim=1)
                        predicted_list.extend(classes.tolist())
                    act_pred_df = pandas.DataFrame({'index' : list(range(len(actual_list))), 'predicted' : predicted_list, 'actual' : actual_list})
                    act_pred_df.drop('index', axis=1).to_csv(folder_name + '/classes_' + str(fold) + '.csv', index=False)
                for index in range(10):
                    os.rename('./graph/data/results/classifier_' + str(index) + '.pt', os.path.join(folder_name + '/' + 'classifier_' + str(index) + '.pt'))
                    
            os.rename('./graph/data/results/classifier.pt', os.path.join(folder_name + '/' + 'classifier.pt'))
            DGL.DataExportToCSV(results, os.path.join(folder_name + '/' + 'results.csv'))
            act_pred_df.drop('index', axis=1).to_csv(folder_name + '/classes.csv', index=False)
            Plotly.ExportToImage(loss_fig, folder_name + '/loss.' + format_type, format=format_type, width='1280', height='720')
            Plotly.ExportToImage(acc_fig, folder_name + '/accuracy.'  + format_type, format=format_type, width='1280', height='720')
            Plotly.ExportToImage(cm_fig, folder_name + '/confusion_matrix.' + format_type, format=format_type, width='1280', height='720')

    elif task.lower() == 'regression':
        loss_labels = ['Epochs', 'Training Loss', 'Validation Loss', 'Testing Loss']
        loss_df = Plotly.DataByDGL(data=results, labels=loss_labels)
        loss_fig = Plotly.FigureByDataFrame(loss_df, labels=loss_labels, title='Root Mean Square Error', xTitle='Epochs', xSpacing=10.0, ySpacing=10.0, yTitle='RMSE')
        Plotly.Show(loss_fig, renderer='browser')
        regressor = DGL.ModelByFilePath('./graph/data/results/regressor.pt')
        predicted = DGL.ModelPredict(regressor, datasets['test_ds'])
        actual = datasets['test_ds'].labels.tolist()
        act_pred_df = pandas.DataFrame({'index' : list(range(len(actual))), 'predicted' : predicted, 'actual' : actual})
        pred_fig = Plotly.FigureByDataFrame(act_pred_df, labels=['index', 'predicted', 'actual'], width=950, height=500, title='Testing', xTitle='Index', xSpacing=200, yTitle='Value', ySpacing=100.0, useMarkers=False, chartType='Line')
        Plotly.Show(pred_fig, renderer='browser')
        corr_fig = Plotly.FigureByDataFrame(act_pred_df, labels=['actual', 'predicted'], width=950, height=500, title='Correlation', xTitle='Actual', xSpacing=100, yTitle='Predicted', ySpacing=100.0, useMarkers=False, chartType='Scatter')
        Plotly.Show(corr_fig, renderer='browser')
        if export:
            rg = 'regression'
            rmse = 'rmse_' + str(round(results['Testing Loss'][0][0], 3)).replace('.', '_')
            ep = 'ep_' + str(results['Epochs'][-1])
            ba = 'ba_' + str(results['Batch Size'][0])
            lr = 'lr_' + str(results['Learning Rate'][0])
            sz = 'sz_' + str(len(DGL.DatasetLabels(datasets['train_ds'])))
            folder_name = './graph/data/results/' + rg + '/' + rmse + '_' + ep + '_' + ba + '_' + lr + '_' + sz
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            if cv_type.lower() == 'k-fold':
                for fold, dataloader in enumerate(dataloaders):
                    regressor = DGL.ModelByFilePath('./graph/data/results/regressor_' + str(fold) + '.pt')
                    actual_list = []
                    predicted_list = []
                    for batched_graph, labels in dataloader:
                        actual_list.extend(labels.tolist())
                        predicted = regressor(batched_graph, batched_graph.ndata['node_attr'].float())
                        predicted_list.extend(predicted.flatten().tolist())
                    act_pred_df = pandas.DataFrame({'index' : list(range(len(actual_list))), 'predicted' : predicted_list, 'actual' : actual_list})
                    act_pred_df.drop('index', axis=1).to_csv(folder_name + '/values_' + str(fold) + '.csv', index=False)
                for index in range(10):
                    os.rename('./graph/data/results/regressor_' + str(index) + '.pt', os.path.join(folder_name + '/' + 'regressor_' + str(index) + '.pt'))
                    
            os.rename('./graph/data/results/regressor.pt', os.path.join(folder_name + '/' + 'regressor.pt'))
            DGL.DataExportToCSV(results, os.path.join(folder_name + '/' + 'results.csv'))
            act_pred_df.drop('index', axis=1).to_csv(folder_name + '/values.csv', index=False)
            Plotly.ExportToImage(loss_fig, folder_name + '/loss.'  + format_type, format=format_type, width='1280', height='720')
            Plotly.ExportToImage(pred_fig, folder_name + '/prediction.'  + format_type, format=format_type, width='1280', height='720')
            Plotly.ExportToImage(corr_fig, folder_name + '/correlation.'  + format_type, format=format_type, width='1280', height='720')
