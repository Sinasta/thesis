# Topological Graphs in Architecture

This is the official repository for the thesis '*[Topological Graphs in Architecture](https://github.com/Sinasta/thesis/blob/main/documentation/thesis.pdf)*' by [Raban Ohlhoff](https://raban-ohlhoff.com/), which was written as part of the Master's programme in Architecture.

## Abstract
Traditional architectural practices often rely on creative intuition and experience rather than systematic analysis and data-driven decision making. With the increasing availability of computational tools and data science techniques, there is an opportunity to bring mathematical and computer science concepts and methods into the architectural context to challenge traditional practices and oﬀer potential improvements. This thesis explores the application of graph theoretical and topological concepts in architecture and investigates the use of graph machine learning methods in the context of architectural analysis, with a particular focus on energy eﬃciency as a key performance metric. To this end, a synthetic architectural dataset containing geometric, categorical, dimensional, energetic, topological and relational information is generated by integrating various space partitioning algorithms combined with architectural control functions into an automated generation pipeline. Subsequently, a classiﬁcation model and a regression model are trained on the generated knowledge graph dataset to evaluate the prediction and classiﬁcation accuracy in terms of energy eﬃciency. The resulting dataset and the code for generating and training the model will be made publicly available to further research in the ﬁeld of graph machine learning in architectural applications. This research demonstrates the potential of a closer integration of various mathematical concepts and computer science methods into the architectural design and veriﬁcation process, and shows the potential of applying knowledge graphs for the abstraction, representation and analysis of architectural objects.

Requirements:

- Machine Learning:
  - [TopologicPy](https://github.com/Sinasta/topologicpy)
  - [DGL](https://github.com/dmlc/dgl)
  - [PyTorch](https://github.com/pytorch/pytorch)
  - [tqdm](https://github.com/tqdm/tqdm)
  - [NumPy](https://github.com/numpy/numpy)
  - [pandas](https://github.com/pandas-dev/pandas)
- Dataset Generation:
  - [PyVoro](https://github.com/joe-jordan/pyvoro)
  - [SciPy](https://github.com/scipy/scipy)
- Energy Simulation:
  - [OpenStudio](https://github.com/NREL/OpenStudio)