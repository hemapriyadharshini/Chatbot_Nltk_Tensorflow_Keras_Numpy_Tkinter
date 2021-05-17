# Chatbot_Nltk_Tensorflow_Keras_Numpy_Tkinter
_Step1:_ Download above files and store it in your local machine

_Step2:_ Execute below command: Python train_chatbot.py

![image](https://user-images.githubusercontent.com/8421214/118532482-0457ef00-b715-11eb-8e36-db36ef77fbac.png)

_Step3:_ Execute below command: Python Chatgui.py

![image](https://user-images.githubusercontent.com/8421214/118536341-9feb5e80-b719-11eb-8160-34a6f92ad5aa.png)

**Output:**

![image](https://user-images.githubusercontent.com/8421214/118537809-66b3ee00-b71b-11eb-8e31-e359beaf61d6.png)

**Observations:**

**intents.json:**
- This file stores chat patterns. 
- It's easy to update the file, however one has to think through the flow of conversation between human and chatbot to build a robust chatbot app

**words.pkl:**
- This is a pickle file, encoded in a particular file format
- Contains unique lemmatized words from the json file
- The file will be updated automatically as with the changes in the json file contents

**Classes.pkl:**
- This is a pickle file, encoded in a file format which cannot be read readily
- Contains the list of categories
- The file will be updated automatically as with the changes in the json file contents

_PS_: Read contents of pickle file by executing ReadPklFile.ipynb

**Train_Chatbot.py:**
- This contains the code to build and train model
- Python Package used: nltk, pickle, numpy, tensorflow keras, random

**chatbot_model.h5:**
- Output of trained model from train_chatbot.py

**chatgui.py:**
- This deploys model in a gui env
- Packages used: nltk, pickle, numpy, tensorflow keras, tkinter
- tkinter is a useful module for building gui based apps

**Notes:**
- Lemmatisation: process of converting/mapping any word to its root word (Ex: word: Trusting - Lemmatized word: Trust)
- keras.models: Model 
- keras.layers: Neural Network layer
- from keras.models import Sequential: Plain linear stack of layers where each layer has exactly one input tensor and one output tensor.
- from keras.layers import Dense: Create deep layers in the neural network
- from keras.layers import Activation: Activate neural network layer
- from keras.layers import Dropout: Drop out neurons randomly to avoid overfitting
- random.shuffle(training): shuffle features randomly to avoid overfitting
- Total nueral network layers: 3 -First layer with 128 neurons, second layer with 64 neurons and 3rd output layer with the number of neurons that are equal to number of intents to predict output intent with softmax. Significance of no's used in each layers - they are all powers of 2s i.e., matrix multiplications (2x2) which is the heart of deep learning
- model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) #Add 1st dense layer with a Rectified Linear Unit activation. A linear activation function is simply the sum of the weighted input to the node, required as input for any activation function
- model.add(Dropout(0.5)) #create a dropout layer with a 50% chance of setting inputs to zero to avoid overfitting
model.add(Dense(64, activation='relu')) #Add 2nd dense layer with a Rectified Linear Unit activation. A linear activation function is simply the sum of the weighted input to the node, required as input for any activation function
model.add(Dropout(0.5))#create a dropout layer with a 50% chance of setting inputs to zero to avoid overfitting
model.add(Dense(len(train_y[0]), activation='softmax')) #Add 3rd dense layer with Softmax function i.e., a softened version of the argmax function that returns the index of the largest value in a list. The max node array value will be the output of the layer and all other nodes output will be 0
- SGD - Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
- sgd= SGD(learning_rate=0.01,decay=1e-6, momentum=0.9, nesterov=True)
- model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy']) 
#loss = categorical crossentropy - Computes the cross-entropy (Average number of bits required to send the message from distribution A to Distribution B) loss between true labels and predicted labels
#optimizer= SGD - Stochastic Gradient Descent (Default Optimizer) - uses a randomly selected instance from the training data to estimate the gradient. This generally leads to faster convergence, but the steps are noisier because each step is an estimate. Gradient descent refers to the steepest rate of descent down a gradient or slope to minimize the value of the loss function as the machine learning model iterates through more and more epochs
#Metrics = Accuracy - Calculates how often predictions equal labels

**Next Steps:**
Chatbot is a booming field in deep learning space and can be used for commercial benefits. This project has a lot of opportunity for enhancements. If you are trying to learn from this code, try making custom enhancements like adding datetime to the chat window, add 'You typing.../Bot typing...' message to appear in the chat window while typing in the text box, add buttons for categories, incorporate sentimental analysis for bot response, text to speech to text, autoamted spell correction, next word prediction, add emojis, etc.,
