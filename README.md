### Note: pre-Release version. 
This small Library is a collection of classes and functions that help generate synthetic graphs 
from the family of Stochastic Block Models (SBM) and evaluate their characteristics.

The Original motivation was to generate synthetic graphs, that can be used to Benchmark Graph Neural Networks.
It implements the original Stochastic-Block-Model (**SBM**) [Holland et al. (1983)](https://www.sciencedirect.com/science/article/abs/pii/0378873383900217) the Degree-Corrected Stochastic-Block-Model (**DC-SBM**) [Karrer & Newman (2011)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.83.016107), the
Attributed Degree-corrected Stoachastic-Block-model (**ADC-SBM**) [Tsitsulin et al. (2022)](https://arxiv.org/pdf/2204.01376) as well as the Labelled Attributed Degree-corrected Stoachastic-Block-model (**L-ADC-SBM**) [Caratiola & Zogaj (2025)](...).

It further serves as suplementary material for the Paper (comming soon).
Due to the Lack of quality Graph Data, this package provides an easy to use API for generating Graphs that help train Machine Learning Models.

Every class from SBM to LADCSBM is essentially a wrapper around NetworkX and can retrun the attributed Graphs with the 
*.to_Nx()*-method. Additionally, some helper functions for Evaluating and plotting the generated graphs are implemented, to assure the right quality of them.
These function will work only with the instance Graph which inherits from SBM. A abitrary NetworkX Graph wont be excepted.  

