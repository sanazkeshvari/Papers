# ListMAP: Listwise learning to rank as maximum a posteriori estimation.

## About
This paper publish in [Information Processing and Management](https://doi.org/10.1016/j.ipm.2022.102962)

### Abstract

Listwise learning to rank models, which optimize the ranking of a document list, are among the
most widely adopted algorithms for finding and ranking relevant documents to user information
needs. In this paper, we propose ListMAP, a new listwise learning to rank model with prior
distribution that encodes the informativeness of training data and assigns different weights to
training instances. The main intuition behind ListMAP is that documents in the training dataset
do not have the same impact on training a ranking function. ListMAP formalizes the listwise loss
function as a maximum a posteriori estimation problem in which the scoring function must be
estimated such that the log probability of the predicted ranked list is maximized given a prior
distribution on the labeled data. We provide a model for approximating the prior distribution
parameters from a set of observation data. We implement the proposed learning to rank model
using neural networks. We theoretically discuss and analyze the characteristics of the introduced
model and empirically illustrate its performance on a number of benchmark datasets; namely
MQ2007 and MQ2008 of the Letor 4.0 benchmark, Set 1 and Set 2 of the Yahoo! learning to rank
challenge data set, and Microsoft 30k and Microsoft 10K datasets. We show that the proposed
models are effective across different datasets in terms of information retrieval evaluation metrics
NDCG and MRR at positions 1, 3, 5, 10, and 20.

