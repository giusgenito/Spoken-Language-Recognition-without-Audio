# Spoken-Language-Recognition-without-Audio
Spoken Language Recognition without Audio. It's my computer vision project for the exam,The goal of the project is to classify the spoken language in the video without using audio, but only by observing lip movements.
Our dataset consisted of 960 videos in the training set, 160 in the test set, and 160 in the validation set. The dataset contains videos in 8 languages: Italian, English, German, Spanish, Dutch, Russian, Japanese and French. 

Key features: Python, Scikit-learn, Tensorflow, Google Colab, Keras, Seaborn

The idea of the project stems from the fact that each sound corresponds to a specific articulation linked to facial movements.
![image](https://github.com/user-attachments/assets/ed653471-dacc-47d8-839c-7ab455782a19)

## Fase di pre-processing

- RGB Images: Reduction of the number of frames to 150, without skipping frames, resizing videos to 50x50 format, and saving in np.array.
<img width="497" alt="image" src="https://github.com/user-attachments/assets/0d0128a0-b1d3-4a42-b215-c7a5b22ecc66" />
- Gray scale Images: Reduction of the number of frames to 150, without skipping frames, resizing videos to 50x50 format, converting videos to grayscale, and saving in np.array
<img width="148" alt="image" src="https://github.com/user-attachments/assets/2e35c4e1-9f9c-45ac-929a-8c52eef08123" />


## Neural network
### Neural network for RGB images 
- ConvLSTM2D: A recurrent network that combines two networks: Conv2D and LSTM. Specifically, three filters were used
- Dropout: A regularization layer used to reduce overfitting
- Dense Layer: A dense layer with 8 neurons and a softmax activation function calculates the probabilities associated with each class.
<img width="501" alt="image" src="https://github.com/user-attachments/assets/9772f903-5caf-4fa2-8923-dc78f077c8a0" />
### Neural network for GrayScale images
- Conv2D: A non-recurrent network used to extract relevant features from the frames and reduce dimensionality
- Dropout: A regularization layer used to reduce overfitting
- Dense Layer: A dense layer with 8 neurons and a softmax activation function calculates the probabilities associated with each class.
<img width="473" alt="image" src="https://github.com/user-attachments/assets/8e1b5186-1125-4f6c-a331-c9a83274f022" />

## Metrics of Evaluation
### Evaluation of RGB images
![image](https://github.com/user-attachments/assets/0fe620b0-f85f-4355-a91c-c9698b1cf495)
<img width="379" alt="image" src="https://github.com/user-attachments/assets/7c079014-f095-4679-92b8-c292eac58005" />

### Evaluation of GrayScale images
<img width="381" alt="image" src="https://github.com/user-attachments/assets/bc294a47-9dc2-429f-a586-d5360d2e92cf" />

