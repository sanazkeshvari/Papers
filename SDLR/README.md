# A Self-Distilled Learning to Rank

# SDLR is a listwise L2R

The knowledge distillation framework of SDLR. Both teacher and student models train a Listwise Learning to Rank model, which is dependent on an approximation of the feature’s bandwidths. The teacher model estimates the appropriate bandwidth matrix in an iterative manner.

<img width="1315" alt="model1" src="https://github.com/sanazkeshvari/RankingSDLR/assets/48029925/b8e5b9db-679e-4c4e-9994-82b44bbd7751">


# Files of SDLR

ListSD: This file is the loss function that can add to file of model folder in Allrank method in "https://github.com/allegro/allRank"
This is the loss function of techear phase.

ListSDStu is the loss function in student phase witch should add to model folder in Allrank method.


