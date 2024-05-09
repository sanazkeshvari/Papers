Results are reported in terms of <b>NDCG</b> and <b>MRR</b>  for baseline methods and SDLR in `Baselines` and `SDLR` directories.

SDLR Data are reported in Noisy and Not_Noisy settings.

Two different noise settings are reported:
<ul>
  <li>Mean=0 and  variance=0.05 In `Normal Distribution`.</li>
  <li>Mean=0 and  variance= each feature's variance In `Feature base`.</li>
</ul>


Not_Noisy results include three different settings as below:
<ul>
    <li> Data were split with different ratios between the teacher and student in 'Different ratio of Data'. </li>
     <li>All data were provided only to the teacher phase (there is no student phase) in 'Teacher Only'. </li>
     <li>Both teacher and student were trained on all the data in 'Train over 100% of training data'. </li>
</ul>

