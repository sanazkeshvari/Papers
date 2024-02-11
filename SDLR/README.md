# A Self-Distilled Learning to Rank(SDLR)

 A Self-Distillation model is used which the structure of teacher model and student model were fixed, to learn how to rank documents come from webpages in information retrieval. The model contains of Fully Connected layers and Transfer Blocks that has Attention layers. The model's structure and its parameters setting was written in the paper.

The keypoints of this research are:
1. SDLR is a listwise learning to rank(L2R) model which is used a listwise loss function to learn how to rank a list of documents.
2. A Self-Distillation model which is one of the novel approches in knowledge disillation models is used in SDLR.
3. SDLR use both data and the content came from the distribution of data for learn which document is more infomative.
4. Having the content of data by its distribution make SDLR a robust model against normal noise with different rate.
5. SDLR outperformed other baseline methods (in benchmark mothods) and the previous research ListMAP.

The knowledge distillation framework of SDLR. Both teacher and student models train a Listwise Learning to Rank model, which is dependent on an approximation of the feature’s bandwidths. The teacher model estimates the appropriate bandwidth matrix in an iterative manner.



<img width="1315" alt="model1" src="https://github.com/sanazkeshvari/Papers/assets/48029925/823bc2fa-9f9a-461d-a3cd-bfc9a2616ed9">





