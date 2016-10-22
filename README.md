# Emotion Recognition

To train the model from scratch -

1. Download the [Cohn-Kanade](http://www.consortium.ri.cmu.edu/ckagree/) dataset.
2. Create a folder named "data" in the project directory.
3. And paste the dataset with the name "Dataset_images" and emotions with the name "Emotion" in the  above created folder.
4. Finally, open terminal in the project directory and then type in-

`python train.py`

After training or if you directly want to try the trained model -

`python main.py`

Dependencies - 
- Numpy 
- Pandas 
- scikit-learn 
- opencv 

The emotions being classified are Neutral, Anger, Contempt, Disgust, Fear, Happy, Sad, Surprise  
Currently the accuracy on the test data is **53.42%** and live feed is not performing great. 

The size of the dataset is not large enough to classify 8 different emotions, so probably train on less number of classes?

_Currently under development_ 


