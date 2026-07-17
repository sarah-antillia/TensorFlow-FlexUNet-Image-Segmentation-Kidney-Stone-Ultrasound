<h2>TensorFlow-FlexUNet-Image-Segmentation-Kidney-Stone-Ultrasound (2026/07/18)</h2>
<h3>Kidney-Stone-Ultrasound: AI Generated Pseudo Masks Segmentation Challenge</h3>
Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>Kidney-Stone-Ultrasound</b> based on our 
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">TensorFlow-FlexUNet-Image-Segmentation-Model</a> 
(TensorFlow Flexible UNet Image Segmentation Model for Multiclass), 
and a 512x512 pixels PNG 
<a href="https://drive.google.com/file/d/1qZjyCSRFsi0ZKtVeRDlbJbKN9FbqrO5U/view?usp=sharing">
<b>Augmented-Kidney-Stone-Ultrasound-ImageMask-Dataset.zip</b></a> 
(<a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>)
, which was derived by us from <br><br>
<b>Stone</b> subset of 
<a href="https://www.kaggle.com/datasets/imtkaggleteam/kidney-stone-classification-and-object-detection">
<b>Kidney Stone | Classification and Object Detection
</b></a> by Mohamadreza Momeni.<br><br>
<hr>
<b>Actual Image Segmentation for Kidney-Stone-Ultrasound Images of 512x512 pixels </b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks.
<br><br>
<table >
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction:inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test/images/10040.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test/masks/10040.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test_output/10040.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test/images/10894.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test/masks/10894.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test_output/10894.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test/images/10932.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test/masks/10932.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test_output/10932.png" width="320" height="auto"></td>
</tr>
 
</table>
<hr>
<br>
<h3>1  Dataset Citation</h3>
The dataset used here was derived from <br><br>
<b>Stone</b> subset of 
<a href="https://www.kaggle.com/datasets/imtkaggleteam/kidney-stone-classification-and-object-detection">
<b>Kidney Stone | Classification and Object Detection
</b></a> by Mohamadreza Momeni.<br><br>
The following explanation was taken from the above web site.<br><br>
<b>About Dataset</b><br>
<br>
<b>Datasets – Kidney Stone Detection Project</b><br>
This folder contains the raw and processed ultrasound images used in the Kidney Stone Detection Challenge.<br>
<br>
<b>Overview</b><br>
Total Images: 9,416<br>
Normal: 4,414<br>
Stone: 5,002<br>
Resolution: 512 x 512 pixels<br>
Format: JPEG / PNG (depending on export)<br>
Source: Multiple hospitals and scan centers, collected using Samsung ultrasound machines 
(RS85, HS60, RS80A, HS70A). <br><br>
<b>License</b><br>
<a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>
<br>
<br>
<b> Important Notes for Annotators</b><br>
When labeling kidney stones:<br>

Look for bright hyperechoic spots.<br>
Identify acoustic shadows behind stones.<br>
Avoid confusing stones with normal tissue textures.<br>
<br>
<h3>
2 Kidney-Stone-Ultrasound ImageMask Dataset
</h3>
<h3>
2.1 Download ImageMask Dataset
</h3>
 If you would like to train this Kidney-Stone-Ultrasound Segmentation model by yourself,
please down load our dataset <a href="https://drive.google.com/file/d/1qZjyCSRFsi0ZKtVeRDlbJbKN9FbqrO5U/view?usp=sharing">
<b>Augmented-Kidney-Stone-Ultrasound-ImageMask-Dataset.zip</b> 
</a>(<a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>) 
 on the google drive,
expand the downloaded, and put it under <b>./dataset/</b> to be:
<pre>
./dataset
└─Kidney-Stone-Ultrasound
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Kidney-Stone-Ultrasound Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/Kidney-Stone-Ultrasound_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br><br>
<h3>
2.2 Derivation of Kidney-Stone-Ultrasound ImageMask Dataset
</h3>
The folder structure of the original Kidney dataset is the following, but it contains no segmentation (mask) files.
<pre>
./Kidney_dataset
  ├─Normal
  │   ├─Normal_1.JPG
...
  │   └─Normal_4414.JPG
  │   
  └─Stone
        ├─Stone_1.JPG
...
        └─Stone_5002.JPG
</pre>
<b>Step 1</b><br>
We generated a 512x512 pixsels master PNG image files from the JPG files in the <b>Stone</b> subset, not including 
<b>Normal</b> subset.
<br><br>
<b>Step 2</b><br>
We generated the first pseudo masks corresponding to the master images 
by applying an inference (segmentation) method of
the first pretrained model <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Mendeley-Kidney-Stone-Ultrasound">
TensorFlow-FlexUNet-Image-Segmentation-Mendeley-Kidney-Ultrasound
</a> to the master images.
<br><br>
<b>Step 3</b><br>
We generated the first ImageMask Dataset from the pairs of Stone images and the first 
pseudo masks, by excluding all empty black masks and their corresponding images.
<br><br>
<b>Step 4</b><br>
We generated the first Augmented ImageMask Dataset from the first ImageMask Dataset 
by using the following image deformation tools.<br>
<a href="https://github.com/sarah-antillia/Image-Deformation-Tool">Image-Deformation-Tool</a><br>
<a href="https://github.com/sarah-antillia/Image-Distortion-Tool">Image-Distortion-Tool</a><br>
<br>
<b>Step 5</b><br>
We trained a <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">TensorFlowFlexUNet Model</a> 
by using the first Augmented Dataset, and then generated the second pseudo masks corresponding to the master images, 
by applying a segmentation method of the second pretrained FlexUNet model.
<br><br>
<b>Step 6</b><br>
We generated the second ImageMask Dataset from the pairs of Stone images and the second 
pseudo masks, by excluding all empty black masks and their corresponding images.
<br><br>
<b>Step 7</b><br>
We finally generated our own  
<a href="">
<b>Augmented-Kidney-Stone-Ultrasound-ImageMask-Dataset.zip</b></a> 
from the second ImageMask Dataset 
by using the following image deformation tools.<br>
<a href="https://github.com/sarah-antillia/Image-Deformation-Tool">Image-Deformation-Tool</a><br>
<a href="https://github.com/sarah-antillia/Image-Distortion-Tool">Image-Distortion-Tool</a><br>
<br>
<h3>
2.3 Train Sample Images and Masks
</h3>
<b>Train sample images</b><br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train sample masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorflowFlexUNet Model
</h3>
 We trained Kidney-Stone-Ultrasound TensorFlowFlexUNet Model by using the 
<a href="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters=16</b> and a large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large <b>num_layers=8</b> (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
normalization  = False
num_classes    = 2
base_filters   = 16
base_kernels  = (11,11)
num_layers    = 8
dropout_rate   = 0.05
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.0001
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and "dice_coef_multiclass".<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience=10 parameter.
<pre>
[train]
patience      = 10
</pre>
<b>Infer section</b><br>
<pre>
[infer] 
images_dir    = "./mini_test/images/"
output_dir    = "./mini_test_output/"
</pre>
<b>RGB color map</b><br>
rgb color map dict for Kidney-Stone-Ultrasound 1+1 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
;Kidney-Stone-Ultrasound 1+1
rgb_map {(0, 0, 0): 0, (255, 255, 255):1}
</pre>
<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer     = True
epoch_change_infer_dir =  "./epoch_change_infer"
epoch_change_infer     = False
epoch_change_infer_dir =  "./epoch_change_infer"
num_infer_images =  6
</pre>
By using this <b>epoch_change_infer</b> callback, on every epoch_change, the <b>infer</b> method of the 
<a href="./src/TensorFlowFlexModel.py">TensorFlowFlexModel</a> class 
can be called
 for 6 images in <b>mini_test</b> folder specified in <b>tiledinfer</b> section. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 
<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (11,12,13)</b><br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (23,24,25)</b><br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>

<br>
In this experiment, the training process was stopped at epoch 25 by EarlyStoppingCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/asset/train_console_output_at_epoch25.png" width="1024" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound</b> folder, 
and run the following bat file to evaluate TensorflowFlexUNet model for Kidney-Stone-Ultrasound.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py  ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/asset/evaluate_console_output_at_epoch25.png" width="1024" height="auto">
<br><br>Image-Segmentation-Kidney-Stone-Ultrasound

<a href="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to the tiledly split <b>Kidney-Stone-Ultrasound/test</b> was very low, and dice_coef_multiclass 
very high as shown below.
<br>
<pre>
categorical_crossentropy,0.0003
dice_coef_multiclass,0.9998
</pre>
<b>Why was the loss so low and the dice_coef so high in the evaluation scores for the test dataset in this segmentation model? </b><br>
The main reason is that the number of black pixels in the Background class is overwhelmingly larger than that of 
white pixels in the Stone class in almost all annotation (mask) data. 
As a result, the Background would be better recognized than the Stone in this multiclass FlexUNet model.
<br>
<h3>5 Inference</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound</b> folder, and run the following bat file to infer segmentation regions for images by the Trained-TensorflowFlexUNet model for Kidney-Stone-Ultrasound.<br>
<pre>
>./3.infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for Kidney-Stone-Ultrasound of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar 
to the ground truth masks, except the fourth case.
<br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test/images/10009.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test/masks/10009.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test_output/10009.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test/images/10109.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test/masks/10109.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test_output/10109.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test/images/10338.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test/masks/10338.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test_output/10338.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test/images/10564.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test/masks/10564.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test_output/10564.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test/images/10894.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test/masks/10894.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test_output/10894.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test/images/distorted_0.01_rsigma0.5_sigma40_10216.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test/masks/distorted_0.01_rsigma0.5_sigma40_10216.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kidney-Stone-Ultrasound/mini_test_output/distorted_0.01_rsigma0.5_sigma40_10216.png" width="320" height="auto"></td>
</tr>

</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. KSSD2025: A New Annotated Dataset for Automatic Kidney Stone Segmentation and Evaluation With Modified U-Net Based Deep Learning Models</b><br>
Murillo F. Murillobouzon; Paulo H. S. de Santana; Gabriel N. Missima; Weverson S. Pereira; Fernando P. Rivera; Gilson A. Giraldi<br>
<a href="https://ieeexplore.ieee.org/document/11165055">https://ieeexplore.ieee.org/document/11165055</a>
<br><br>
<b>2. A deep learning system for automated kidney stone detection and volumetric segmentation on noncontrast CT scans</b><br>
Daniel C. Elton, Evrim B.Turkbey, Perry J.Pickhardt, Ronald M.Summers<br>
<a href="https://www.moreisdifferent.com/assets/my_papers/B_AI_medical_imaging/2022_Z_Elton_Medical_Physics_kidney_stone_detector.pdf">
https://www.moreisdifferent.com/assets/my_papers/B_AI_medical_imaging/2022_Z_Elton_Medical_Physics_kidney_stone_detector.pdf</a>
<br><br>
<b>3. TensorFlow-FlexUNet-Image-Segmentation-KSSD2025-Kidney-Stone-CT</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-KSSD2025-Kidney-Stone-CT">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-KSSD2025-Kidney-Stone-CT
</a>
<br>
<br>
<b>4. TensorFlow-FlexUNet-Image-Segmentation-Mendeley-Kidney-Stone-Ultrasound </b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Mendeley-Kidney-Stone-Ultrasound">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Mendeley-Kidney-Stone-Ultrasound
</a>
<br>
<br>
<b>5. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
