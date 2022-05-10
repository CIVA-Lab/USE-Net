# USE-Net: Extending U-Net Network for Improved Nuclei Instance Segmentation Accuracy in Histopathology Images
Analysis of morphometric features of nuclei plays an important role in understanding disease progression and predict efficacy of treatment. First step towards this goal requires segmentation of individual nuclei within the imaged tissue. Accurate nuclei instance segmentation is one of the most challenging tasks in computational pathology due to broad morphological variances of individual nuclei and dense clustering of nuclei with indistinct boundaries. It is extremely laborious and costly to
annotate nuclei instances, requiring experienced pathologists to manually draw the contours, which often results in the lack of annotated data. Inevitably subjective annotation and mislabeling prevent supervised learning approaches to learn from accurate samples and consequently decrease the generalization capacity to robustly segment unseen organ nuclei, leading to over- or under-segmentations as a result. To address these issues, we use a variation of U-Net that uses squeeze and excitation blocks (USE-Net) for robust nuclei segmentation. The squeeze and excitation blocks allow the network to perform feature recalibration by emphasizing informative features and suppressing less useful ones. Furthermore, we extend the proposed network USE-Net not to generate only a segmentation mask, but also to output shape markers to allow better separation of nuclei from each other particularly within dense clusters. The proposed network was trained, tested, and evaluated on 2018 MICCAI Multi-Organ-Nuclei-Segmentation (MoNuSeg) challenge dataset. Promising results were obtained on unseen data despite that the data used for training USE-Net was significantly small.

<br/>

## Methods
The overall flow of the proposed pipeline is as follows. Initially, the proposed pipeline begins with a preprocessing step to reduce color variations in input images. Afterward, the preprocessed data is augmented and fed into the network. Later, the output masks from the network are post-processed using mathematical morphology. Finally, the watershed method is applied to separate attached nuclei from each other and get the labeled mask as a final output from pipeline, where each nuclei has its unique id.

<br/>

![](/figures/pipeline.png)

<br/>

### A. Preprocessing
By increasing the input samples and reducing the color variation in the training images, our network will be able to produce more accurate segmentation results. The color variation in histology images can be reduced by using color normalization techniques. Moreover, different data augmentation strategies are used during training to improve the network’s generalizability.

Structure preserving color normalization (SPCN) is commonly used for color normalization of histology images, where the structure of the image is preserved while the color is normalized to a common color space. Consequently, the learning process for segmentation is streamlined.

<br/>

![](/figures/SPCN_exp.png)

<br/>

### B. Nuclei Segmentation Network
The used encoder-decoder deep learning network architectureUSE-Net, is similar to the state-of-the-art network architecture U-Net, where in the encoder side the
SE-ResNet-50 is used as backbone instead of normal UNet encoder, and squeeze and excitation blocks are used after each residual block of the ResNet-50.

<br/>

![](/figures/USE_Net_arch.png){ width=80% }

<br/>

### C. Postprocessing
USE-Net gives segmentation mask and shape marker as the class label probabilities. Thresholding those probabilities leads to foreground/background segmentation mask and shape marker. To remove small spurious detection and fill the gap morphology operations are applied in the segmentation mask. After thresholding shape marker, connected component is applied to it, since it has better nuclei separation. Obtained labeled shape marker map along with the binarized mask are given as inputs to the watershed algorithm to generate a final instance segmentation of the nuclei.

<br/>

![](/figures/qualitative_results.png){ width=80% }

<br/>

```More details on a paper:``` https://ieeexplore.ieee.org/document/9762213/ 

<br/>

**The codes and details about how to train and inference the proposed USE-Net network will be uploaded soon.** 

<br/>

## Project Collaborators and Contact

**Author:** Gani Rahmon, Imad Eddine Toubal and Kannappan Palaniappan

Copyright &copy; 2022-2023. Gani Rahmon, Imad Eddine Toubal and Dr. K. Palaniappan and Curators of the University of Missouri, a public corporation. All Rights Reserved.

**Created by:** Ph.D. students: Gani Rahmon and Imad Eddine Toubal
Department of Electrical Engineering and Computer Science,  
University of Missouri-Columbia  

For more information, contact:

* **Gani Rahmon**  
226 Naka Hall (EBW)  
University of Missouri-Columbia  
Columbia, MO 65211  
gani.rahmon@mail.missouri.edu  

* **Imad Eddine Toubal**  
226 Naka Hall (EBW)  
University of Missouri-Columbia  
Columbia, MO 65211   
itoubal@mail.missouri.edu

* **Dr. K. Palaniappan**  
205 Naka Hall (EBW)  
University of Missouri-Columbia  
Columbia, MO 65211  
palaniappank@missouri.edu
