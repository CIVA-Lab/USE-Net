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

![](/figures/SPCN-exp.png)

<br/>

### B. Nuclei Segmentation Network
The used encoder-decoder deep learning network architectureUSE-Net, is similar to the state-of-the-art network architecture U-Net, where in the encoder side the
SE-ResNet-50 is used as backbone instead of normal UNet encoder, and squeeze and excitation blocks are used after each residual block of the ResNet-50.

<br/>

![](/figures/USE-Net-arch.png)

<br/>

### C. Postprocessing
USE-Net gives segmentation mask and shape marker as the class label probabilities. Thresholding those probabilities leads to foreground/background segmentation mask and shape marker. To remove small spurious detection and fill the gap morphology operations are applied in the segmentation mask. After thresholding shape marker, connected component is applied to it, since it has better nuclei separation. Obtained labeled shape marker map along with the binarized mask are given as inputs to the watershed algorithm to generate a final instance segmentation of the nuclei.

<br/>

![](/figures/qualitative-results.png)

<br/>

```More details on a paper:``` https://ieeexplore.ieee.org/document/9762213/ 

<br/>

# How to use proposed pipeline
Proposed pipeline has three stages: 

    1. Preprocessing

    2. Deep Learning

    3. Postprocessing

<br/>

## 1. Preprocessing

**SPCN** folder contains all scripts used to do preprocessing. The preprocessing is done in ```MATLAB```

<br/>

### A. Conversion from XML to Images 

<br/>

Converting ground truth (GT) and getting mask and shape marker from GT.

1. Put MoNuSeg train data in a folder called ```MoNuSeg_TrainingData```, and MoNuSeg test data in a folder called ```MoNuSegTestData```. 

2. Change input path and extensions accordingly in ```convertXMLToImgTrain.m``` and run the script to convert train data.

3. Change input path and extensions accordingly in ```convertXMLToImgTest.m``` and run the script to convert test data.

4. The new folders will be created and the converted files will be saved in those folders.

<br/>

### B. Running Color Normalization

<br/>

For color normalization SPCN is used. 

The following repository is used to run SPCN : https://github.com/abhishekvahadane/CodeRelease_ColorNormalization 

As a target the following training image of liver tissue ```TCGA-B0-5711-01Z-00-DX1```

<br/>

## 2. USE-Net (Deep Learning)

**USE-Net** folder contains all scripts used to do training and inference. The deep learning is done in ```Python```

<br/>

### A. Trainig

<br/>

To train USE-Net data need to be in a correct folder.

1. Put converted train data in a folder called ```trainData``` inside ```dataset``` folder
    ```
    trainData/
        BinaryMask/
        Inputs/
        Marker/
        SPCN/    
    ```

2. Create folder named  ```augTrainData``` with same structure as ```trainData```
    ```
    augTrainData/
        BinaryMask/
        Inputs/
        Marker/
        SPCN/    
    ```

3. Run ```DataAugmentation.py``` script to augment data and store it in ```augTrainData``` folder.

4. Change input, label, marker paths and give a model name in ```TrainUSENet.py``` and run the script to train the network. It requires ```se_resnet50``` as a backbone.

5. The trained model will be saved in the folder named ```models```. The weights used in the paper is provided inside folder ```models```.

<br/>

### B.Inference

<br/>

To run inference on the test data the data needs to be in a correct folder and the model used for inference needs to be in ```models``` folder.

1. Put converted test data in a folder called ```testData``` inside ```dataset``` folder
    ```
    testData/
        BinaryMask/
        Inputs/
        Marker/
        SPCN/    
    ```

2. Change input paths and give a pretrained model name in ```InferUSENet.py``` and run the script to make inference.

3. The results will be saved in a folder named ```output```. 
    ```
    output/
        Mask/
        Marker/ 
    ```

<br/>

## 3. Postprocessing

**Postprocessing** folder contains all scripts used to do postprocessing. The postprocessing is done in ```MATLAB```.

<br/>

### A. Running Postprocessing

<br/>

After obtaining mask and marker, the postprocessing is performed using ```watershed``` algorithm to get final labels. 

1. Put mask and marker outputs from USE-Net in a folder called ```USE-Net-Final```
    ```
    USE-Net-Final/
        Mask/
        Marker/   
    ```

2. Change input path accordingly in ```postProcessing.m``` and run the script to get final labels for evaluation.

3. The final labels will be saved in a folder named ```Label``` inside ```USE-Net-Final``` folder.
    ```
    USE-Net-Final/
        Label/
        Mask/
        Marker/   
    ```

### B. Evaluation

<br/>

After obtaining final labels, the average aggregated Jaccard Index (AJI) is computed to evaluate the performance. 

1. Create a folder named ```BinaryLabeled``` inside folder ```TestGT```  and place the test data converted Ground Truth inside that folder.

2. Change input path accordingly in ```computeAverageAJI.m``` and run the script to get evaluation results, such as:
  
    a. Individual AJI

    b. Nuclei Counts (correct, missing)

    c. Mean AJI

3. Observe the mean AJI.

<br/>

<br/>

## Project Collaborators and Contact

**Author:** Gani Rahmon, Imad Eddine Toubal and Kannappan Palaniappan

Copyright &copy; 2022-2023. Gani Rahmon, Imad Eddine Toubal and Dr. K. Palaniappan and Curators of the University of Missouri, a public corporation. All Rights Reserved.

**Created by:** Ph.D. student: Gani Rahmon  
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
