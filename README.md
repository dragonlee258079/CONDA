# CONDA
Code release for the ECCV 2024 paper "CONDA: Condensed Deep Association Learning for Co-Salient Object Detection" by Long Li, Junwei Han*, Dingwen Zhang, Zhongyu Li, Salman Khan, Rao Anwer, Hisham Cholakkal, Nian Liu*, Fahad Shahbaz Khan

<div style="text-align: center;">  
  <img src="./introduction.jpg" alt="alt_text" width="700">  
</div>  

<img src="./introduction.jpg" alt="alt_text" style="display: block; margin-left: auto; margin-right: auto; width: 800px;"> 

## Abstract
Inter-image association modeling is crucial for co-salient object detection. Despite satisfactory performance, previous methods still have limitations on sufficient inter-image association modeling. Because most of them focus on image feature optimization under the guidance of heuristically calculated raw inter-image associations. They directly rely on raw associations which are not reliable in complex scenarios, and their image feature optimization approach is not explicit for inter-image association modeling. To alleviate these limitations, this paper proposes a deep association learning strategy that deploys deep networks on raw associations to explicitly transform them into deep association features. Specifically, we first create hyperassociations to collect dense pixel-pair-wise raw associations and then deploys deep aggregation networks on them. We design a progressive association generation module for this purpose with additional enhancement of the hyperassociation calculation. More importantly, we propose a correspondence-induced association condensation module that introduces a pretext task, i.e. semantic correspondence estimation, to condense the hyperassociations for computational
burden reduction and noise elimination. We also design an object-aware cycle consistency loss for high-quality correspondence estimations. Experimental results in three benchmark datasets demonstrate the remarkable effectiveness of our proposed method with various training settings.

## Result
The prediction results of our dataset can be download from [prediction](https://pan.baidu.com/s/1vS3d0Jk0PygoL2FbHuM69Q?pwd=g5hd) (g5hd).

<img src="./quantitative_result.jpg" alt="alt_text" width="700">  
<img src="./qualitative_result.jpg" alt="alt_text" width="700">  

