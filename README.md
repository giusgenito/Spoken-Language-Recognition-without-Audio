# Spoken-Language-Recognition-without-Audio
Spoken Language Recognition without Audio. It's my computer vision project for the exam,The goal of the project is to classify the spoken language in the video without using audio, but only by observing lip movements.
Our dataset consisted of 960 videos in the training set, 160 in the test set, and 160 in the validation set. The dataset contains videos in 8 languages: Italian, English, German, Spanish, Dutch, Russian, Japanese and French. 

The idea of the project stems from the fact that each sound corresponds to a specific articulation linked to facial movements.
![image](https://github.com/user-attachments/assets/ed653471-dacc-47d8-839c-7ab455782a19)

## Fase di pre-processing
Reduction of the number of frames to 150, without skipping frames, resizing videos to 50x50 format, and saving in np.array.

<img width="497" alt="image" src="https://github.com/user-attachments/assets/0d0128a0-b1d3-4a42-b215-c7a5b22ecc66" />


## Neural network
- ConvLSTM2D: A recurrent network that combines two networks: Conv2D and LSTM. Specifically, three filters were used
- Dropout: A regularization layer used to reduce overfitting
- Dense Layer: A dense layer with 8 neurons and a softmax activation function calculates the probabilities associated with each class.
<img width="501" alt="image" src="https://github.com/user-attachments/assets/9772f903-5caf-4fa2-8923-dc78f077c8a0" />
