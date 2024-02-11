# A Self-Distilled Learning to Rank

 A Self-Distillation model is used which the structure of teacher model and student model were fixed, to learn how to rank webpages in information retrieval. The model contains of Fully Connected layers and Transfer Blocks that has Attention layers. the models structure and their parameters setting was written in the paper.

## SDLR is a listwise L2R

The knowledge distillation framework of SDLR. Both teacher and student models train a Listwise Learning to Rank model, which is dependent on an approximation of the feature’s bandwidths. The teacher model estimates the appropriate bandwidth matrix in an iterative manner.



<img width="1315" alt="model1" src="https://github.com/sanazkeshvari/Papers/assets/48029925/823bc2fa-9f9a-461d-a3cd-bfc9a2616ed9">





