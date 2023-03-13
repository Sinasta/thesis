# thesis

The first part of this project aims to generate a synthetic dataset of apartment floor plans using parametric tools and layout algorithms to generate various layouts within a defined range of variability.

Apartments are generated through parametric geometry subdivision using Python with various algorithms such as Voronoi diagrams, KD trees, and treemapping algorithms. The generated data is analyzed from various geometric aspects using the Python library Topologic. Openings such as doors and windows are integrated into the basic geometry in a variable pattern within the framework of defined spatial and architectural rules. The data is then enhanced to a three-dimensional model and stored in JSON files. Finally Energy performance simulation is performed for each apartment using Energy+ and Openstudio.

The evaluated synthetic data can be used as training data for a machine learning model that predicts the energy performance class of the architectural objects using the topological graphs and their added dictionaries. The focus of this work is on exploring the potential of graph machine learning algorithms in combination with architectural data.

The ongoing research for this thesis delves into the field of architectural design and energy efficiency, with a specific focus on the use of graph representation in architectural design and the application of machine learning techniques to classify buildings based on their energy performance. The study aims to contribute to the advancement of knowledge in the field of sustainable building design practices by utilizing a graph-based approach to architectural object representation and machine learning for building energy performance classification.

The dataset is a graphical dataset that contains energy consumption data for different building layouts. The different node labels in those graphs include information such as different room types in the building, including utility, livingroom, bedroom, bathroom, and toilet. The rooms can be of different sizes. Furthermore the nodes can be of the type window and have different sizes, as well as being further categorized based on their orientation such as _ne, _s, _nw, etc.

The target variable in the dataset is the "total site energy consumption per surface area" of the corresponding building, which is measured in units of MJ/m2.

For the sake of comparison, the classification problem has been addressed using eight different models: RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier, KNeighborsClassifier, SVC, LogisticRegression, MLPClassifier, and DGLClassifier. The models have been trained and tested on a dataset consisting of 4160 instances and five different classes.

From the classification report, we can observe that the models have varying levels of performance on the dataset. The best-performing models in terms of accuracy are LogisticRegression and SVC with 80% and 77% accuracy, respectively. These models have the highest precision, recall, and f1-score across all five classes.

Further investigation about the utility of graph machine learning algorithms will be conducted and evaluated.
