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
    dataset = build_dataset(amount)
    results = train(dataset, batch_size=50, epochs=100, lr=0.01, split=0.2)
    return results, dataset

if __name__ == "__main__":
    results, dataset = main(amount=int(input('amount: ')))
    
    DGL.Show(results,
             labels=['Epochs', 'Training Accuracy', 'Testing Accuracy', 'Training Loss', 'Testing Loss'],
             title='Results',
             x_title='Epoch',
             x_spacing=1.0,
             y_title='',
             y_spacing=0.1,
             use_markers=False,
             chart_type='Line',
             renderer='browser')
    dist = DGL.CategoryDistribution(labels=dataset.labels.tolist(), categories=None, mantissa=4)
    pie = Plotly.FigureByPieChart(dist['ratios'][0], dist['categories'][0])
    Plotly.Show(pie, camera=[0, 0, 4], renderer='browser', target=[0, 0, 0], up=[0, 1, 0])
    classifier = DGL.ClassifierBypath('./graph/data/classifier/classifier.pt')
    predicted = DGL.Predict(dataset, classifier)['labels']
    cm = Plotly.FigureByConfusionMatrix(DGL.ConfusionMatrix(dataset.labels.tolist(), predicted, list(range(5))), list(range(5)))
    Plotly.Show(cm, camera=[0, 0, 4], renderer='browser', target=[0, 0, 0], up=[0, 1, 0])
