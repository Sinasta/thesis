from generation.tools import *
from generation.outline import *
from generation.algorithms.treemap import *
from generation.algorithms.voronoi_treemap import *
from generation.algorithms.voronoi import *

def main(amount:int = 2, seed:int = None):
    for i in range(1, amount):
        if seed:
            seed += 1 + i
        else:
            seed = np.random.randint(9**9)
        outline = pick_outline_method(seed=seed)
        room_amount = room_amount_from_total_area(outline)
        room_sizes = create_rooms(room_amount, seed=seed)
        names_sorted = get_sorted_names(room_sizes)
        percentages_list = normalize(room_sizes)
        ratio_tree_dict = build_tree_list(percentages_list, seed=seed)
        layout, layout_type = pick_layout_method(seed, outline, ratio_tree_dict)
        layout = add_room_names(layout, names_sorted)
        check_if_layout_good(layout)
        print(draw(layout, show=False, export=True, name=str(seed) + '.png'))

if __name__ == "__main__":
    main()
