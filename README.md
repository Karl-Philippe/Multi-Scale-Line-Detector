# Multi-Scale Line Detector

## Introduction

# Multi-Scale Line Detector

## Introduction

# Multi-Scale Line Detector

## Introduction

The Multi-Scale Line Detector project was developed as part of the **GBM8770: Traitement numérique d’images médicales** course at Polytechnique Montréal during the Autumn 2020 semester. The project involved the implementation of a sophisticated line detection algorithm designed for medical image segmentation, with a particular focus on retinal vessel detection.

### Key Accomplishments:

1. **Implementation of the Multi-Scale Line Detector (MSLD)**: The core of the project was the successful development of an MSLD algorithm, which applies convolution operations to detect lines at various scales and orientations within medical images. The implementation optimized the algorithm's efficiency by precomputing masks for average intensity calculations over different window sizes and line lengths.

2. **Segmentation and Threshold Learning**: To improve segmentation accuracy, the project incorporated a dynamic thresholding method, leveraging ROC curves to identify the optimal threshold that maximizes precision. This method was validated against a standard dataset (DRIVE) used for training and testing.

3. **Performance Evaluation and Analysis**: The project involved a thorough evaluation of the algorithm's performance, utilizing metrics such as precision, ROC curves, and the Sørensen-Dice coefficient. Both global and local segmentation metrics were computed to assess the algorithm's effectiveness in different regions of the images.

4. **Experimental Validation**: The project included designing and conducting experiments to validate the research hypotheses presented in the referenced article. These experiments provided insights into the algorithm's performance and the influence of various hyperparameters on segmentation quality.

5. **Discussion and Recommendations**: The final phase of the project included an in-depth discussion on the choice of hyperparameters, the impact of different thresholding techniques, and recommendations for further improvements. This analysis was based on the experimental results and aimed at enhancing the MSLD algorithm's applicability to medical image analysis.

The project's outcome demonstrated the potential of the MSLD algorithm in medical image segmentation, offering a solid foundation for further research and development in the field of digital image processing.


## Installation

To set up the required dependencies for the Multi-Scale Line Detector project, follow these steps:

1. **Ensure you are in the project's root directory**:

   ```bash
   cd path/to/Multi-Scale-Line-Detector
   ```

2. **Install the required packages**:

    ```bash
    pip install -r requirements.txt
    ```

    This command will install all necessary dependencies specified in the requirements.txt file.

