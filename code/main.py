from generation.tools import *
from generation.outline import *
from generation.algorithms.treemap import *
from generation.algorithms.voronoi_treemap import *
from generation.algorithms.voronoi import *
from energy.add_apertures_energy_sim import *
from graph.tools import *

import os

import topologicpy
import topologic

from topologicpy.DGL import DGL
from topologicpy.Plotly import Plotly
import pandas as pd

def main(amount, seed = 1):
    for frame in tqdm(range(amount), desc='computing graphs'):
        if os.path.exists('./graph/data/graph_batches/graph_batch_' + str(frame) + '.bin'):
            continue
        if seed:
            seed += 100 * (frame + 1)
        else:
            seed = np.random.randint(9**9)
        done = False
        while not done:
            try:
                outline = pick_outline_method(seed=seed)
                room_amount = room_amount_from_total_area(outline)
                room_sizes = create_rooms(room_amount, seed=seed)
                names_sorted = get_sorted_names(room_sizes)
                percentages_list = normalize(room_sizes)
                ratio_tree_dict = build_tree_list(percentages_list, seed=seed)
                layout, layout_type = pick_layout_method(seed, outline, ratio_tree_dict)
                layout = add_room_names(layout, names_sorted)
                done = check_if_layout_good(layout)
                if not done:
                    seed += 1
                    continue
                ccomplex_database = add_apertures_energy_sim(layout)
            except RuntimeError:
                done = False
                seed += 1
        save_graph_batch(ccomplex_database, str(frame))
        save_geometry_batch(ccomplex_database, str(frame))
    unbalanced_dataset = build_dataset(amount)
    dataset = DGL.BalanceDataset(unbalanced_dataset, unbalanced_dataset.labels.tolist(), method="undersampling")
    results = train(dataset, batch_size=50, epochs=100, lr=0.01, split=0.2, lr_decay=0.9)
    return results, dataset

if __name__ == "__main__":
    results, dataset = main(amount=int(input('amount: ')))
    loss_fig = DGL.Show(results, labels=['Epochs', 'Training Loss', 'Testing Loss'], title='Loss', x_title='Epoch', y_title='Loss', renderer='browser')
    Plotly.ExportToImage(loss_fig, './graph/data/results/loss.svg', format='svg', width='1280', height='720')
    acc_fig = DGL.Show(results, labels=['Epochs', 'Training Accuracy', 'Testing Accuracy'], title='Accuracy', x_title='Epoch', y_title='Accuracy', renderer='browser')
    Plotly.ExportToImage(acc_fig, './graph/data/results/accuracy.svg', format='svg', width='1280', height='720')
    labels = dataset.labels.tolist()
    #dist = DGL.CategoryDistribution(labels, categories=None, mantissa=4)
    #dist_fig = Plotly.FigureByPieChart(dist['ratios'][0], dist['categories'][0])
    #Plotly.Show(dist_fig, renderer='browser')
    #Plotly.ExportToImage(dist_fig, './graph/data/results/distribution.svg', format='svg', width='1280', height='720')
    classifier = DGL.ClassifierByFilePath('./graph/data/classifier/classifier.pt')
    predicted = DGL.Predict(dataset, classifier)['labels']
    cm = DGL.ConfusionMatrix(labels, predicted, normalize=True)
    cm_fig = Plotly.FigureByConfusionMatrix(cm, list(range(5)), showScale=False)
    Plotly.Show(cm_fig, renderer='browser')
    Plotly.ExportToImage(cm_fig, './graph/data/results/confusion_matrix.svg', format='svg', width='1280', height='720')
    accuracy = DGL.Accuracy(labels, predicted)
    #print('Duration: ', results['Duration'][0])
    #print('Optimizer: ', results['Optimizer'][0])
    #print('CV Type: ', results['CV Type'][0])
    #print('Split: ', results['Split'][0])
    #print('K-Folds: ', results['K-Folds'][0])
    #print('HL Widths: ', results['HL Widths'][0][0])
    #print('Conv Layer Type: ', results['Conv Layer Type'][0])
    #print('Pooling: ', results['Pooling'][0])
    #print('Learning Rate: ', results['Learning Rate'][0])
    #print('Batch Size: ', results['Batch Size'][0])
    #print('Accuracy: ', accuracy['accuracy'])
    #print('Size: ', accuracy['size'])
    #print('Correct: ', accuracy['correct'])
    #print('Wrong: ', accuracy['wrong'])
    df = pd.DataFrame([accuracy['size'], accuracy['correct'], accuracy['wrong']], ['Size', 'Correct', 'Wrong'], ['Metric'])
    acc_info_fig = Plotly.FigureByDataFrame(df, labels=['Metric', 'Size', 'Correct', 'Wrong'], title='Accuracy: ' + str(accuracy['accuracy']), x_title='Amount', x_spacing=200.0, y_title='', chart_type='Bar')
    Plotly.Show(acc_info_fig, renderer='browser')
    Plotly.ExportToImage(acc_info_fig, './graph/data/results/accuracy_info.svg', format='svg', width='1280', height='720')
