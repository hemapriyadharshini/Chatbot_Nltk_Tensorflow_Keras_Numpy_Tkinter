import nltk #Import Natural Language tool Kit library
from nltk.stem import WordNetLemmatizer #Converts words to their root words Ex: Believing to Belief
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras import *
from keras.models import Sequential # plain stack of layers where each layer has exactly one input tensor and one output tensor.
from keras.layers import Dense # Create deep layers in the neural network
from keras.layers import Activation #Activate neural network layer
from keras.layers import Dropout # Drop out neurons randomly to avoid overfitting
#from keras.optimizers import SGD
import random #Select random neurons

#Initialize variables
words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)


for i in intents['intents']:
    for pattern in i['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add words to the main pool
        documents.append((w, i['tag']))

        # add to our classes list
        if i['tag'] not in classes:
            classes.append(i['tag'])

# lemmaztize, lower each word, remove duplicates and sort words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents\n")
# classes
print (len(classes), "classes:\n", classes, '\n')
# words = all words, vocabulary
print (len(words), "unique lemmatized words:\n", words,"\n")


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


training = [] # create training data
output_empty = [0] * len(classes) # create an empty array for output
# training set, collect bag of words from each sentence
for doc in documents:
    bag = []  # initialize our bag of words
    pattern_words = doc[0]  # list of tokenized words for the pattern
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words] # lemmatize each word - create base word, in order to represent related words
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0) # create our bag of words array with 1, if word match found in current pattern
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row]) #Append all bag of words as one training set

random.shuffle(training) # shuffle our features randomly 
training = np.array(training) #Assign randomly shuffled training set as an np.array
# create train and test lists. X - patterns i.e., understand the conversation flow of the user, Y - intents i.e., task or action that the chatbot user wants to perform
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


# Create model - 3 layers. 
#First layer with 128 neurons, second layer with 64 neurons and 3rd output layer with the number of neurons that are equal to number of intents to predict output intent with softmax. Significance of no's used in each layers - they are all powers of 2s i.e., matrix multiplications (2x2) which is the heart of deep learning
model = Sequential() #Create 3 linear stack of layers
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) #Add 1st dense layer with a Rectified Linear Unit activation. A linear activation function is simply the sum of the weighted input to the node, required as input for any activation function
model.add(Dropout(0.5)) #create a dropout layer with a 50% chance of setting inputs to zero to avoid overfitting
model.add(Dense(64, activation='relu')) #Add 2nd dense layer with a Rectified Linear Unit activation. A linear activation function is simply the sum of the weighted input to the node, required as input for any activation function
model.add(Dropout(0.5))#create a dropout layer with a 50% chance of setting inputs to zero to avoid overfitting
model.add(Dense(len(train_y[0]), activation='softmax')) #Add 3rd dense layer with Softmax function i.e., a softened version of the argmax function that returns the index of the largest value in a list. The max node array value will be the output of the layer and all other nodes output will be 0

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
#opt= SGD(learning_rate=0.01,decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy']) 
#loss = categorical crossentropy - Computes the cross-entropy (Average number of bits required to send the message from distribution A to Distribution B) loss between true labels and predicted labels
#optimizer= SGD - Stochastic Gradient Descent (Default Optimizer) - uses a randomly selected instance from the training data to estimate the gradient. This generally leads to faster convergence, but the steps are noisier because each step is an estimate. Gradient descent refers to the steepest rate of descent down a gradient or slope to minimize the value of the loss function as the machine learning model iterates through more and more epochs
#Metrics = Accuracy - Calculates how often predictions equal labels

#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=250, batch_size=5, verbose=1)
#np.array(train_x) - Numpy array training patterns
#np.array(train_y) - Numpy array training intents
#epochs - Integer- Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided
#Batchsize - Number of samples per gradient update; Default: 32
#Verbose=1; Default value- logs the training progress of each epochs
model.save('chatbot_model.h5', hist)
print('No of Patterns:',(len(train_x[0])))
print('No of Intents:',(len(train_y[0])))
print("model created")
