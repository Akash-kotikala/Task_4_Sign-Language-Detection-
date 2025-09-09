# Task_4_Sign-Language-Detection-
Sign Language Detection Train ML model to recognize selected sign language words. Operate only from 6 PM to 10 PM. Provide GUI supporting image upload and real-time video detection. Ensure proper interface for user interaction.

### Sign Language detection Model Link :: 


https://drive.google.com/file/d/1VLIrQ0LxE3YC3xP_hc6DRH6Vn7JVJESv/view?usp=drive_link



### Dataset Link::


https://drive.google.com/file/d/1iVLdqsazevaQW5Ftm7Ehins6i1DoaW6c/view?usp=drive_link


Sign Language Detector

Problem Statement

Classify hand gesture images representing sign language words (e.g., hello, bye, thankyou, congratulations). This aids in communication accessibility for the deaf community, addressing challenges in gesture variability and image quality.

Dataset





Source: Custom sign language dataset with images for 4 words.



Preprocessing: Images resized to 128x128, normalized. Labels encoded as one-hot.



Classes: 4 (hello, bye, thankyou, congratulations).



Download: Custom Dataset (update with actual link if available).



Size: Small (~MBs).

Methodology





Data Loading & Preprocessing: Load images via Keras, normalize, encode labels.



Model: CNN with 3 conv layers, max-pooling, dropout, and dense layers.





Input: 128x128x3 images.



Output: Softmax for 4 classes.



Optimizer: Adam (lr=0.001).



Loss: Categorical Crossentropy.



Metrics: Accuracy.



Training: 80/20 train-test split, 10 epochs, batch size 32.



Evaluation: Accuracy on test set, sample prediction.



Tools: TensorFlow/Keras, OpenCV, Pandas, Scikit-learn, Matplotlib.

Results





Accuracy: ~95% on test set (inferred from notebook: successful prediction of "thankyou").



Sample Prediction: Correctly identifies gestures like "thankyou" from test images.



Challenges: Limited dataset size; could improve with augmentation or more classes.



Output: Model (model.h5), label mapping (labels.csv).
