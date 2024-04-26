Technologies and Tools
Python
TensorFlow
Keras
OpenCV

   Process
*Run set_hand_histogram.py to set the hand histogram for creating gestures.

*Once you get a good histogram, save it in the code folder.

*Added gestures and label them using OpenCV which uses webcam feed. by running create_gestures.py and stores them in a database.

*Add different variations to the captured gestures by flipping all the images by using Rotate_images.py.

*Run load_images.py to split all the captured gestures into training, validation and test set.

*To view all the gestures, run display_gestures.py .

*Train the model using Keras by running cnn_model_train.py.

*Run final.py. This will open up the gesture recognition window which will use your webcam to interpret the trained American Sign Language gestures.
