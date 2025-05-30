This small Library is a collection of classes and functions that help generate synthetic graphs 
from the family of Stochastic Block Models (SBM) and evaluate their characteristics.

The Original motivation was to generate synthetic graphs, that can be used to Benchmark Graph Neural Networks.
It extends [Tsitsulin et al. (2022)](https://arxiv.org/pdf/2204.01376) Attributed Degree-corrected Stoachastic Block model by adding node lables, that are correlated to the community-labels and node-feature vectors. 
Due to the Lack of quality Graph Data, this package provides an easy to use API for generating Graphs that help train Machine Learning Models.

Every class from SBM to LADCSBM is essentially a wrapper around NetworkX and can retrun the attributed Graphs with the 
*.to_Nx()*-method. Additionally, some helper functions for Evaluating and plotting the generated graphs are implemented, to assure the right quality of them.
These methods will work only with the instance Graph which inherits from SBM. A abitrary NetworkX Graph wont be excepted.  

