import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from datetime import datetime

from keras.models import load_model
model = load_model('chatbot_model.h5') #Load trained model output from train_chatbot.py as an input to chatgui.py
import json
import random
intents = json.loads(open('intents.json').read()) #Load intents from json file
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence) # tokenize the pattern - split words into array
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]  # stem each word - create short form for word
    return sentence_words #return stem word Ex: if Believing, then setence_words = Belief 

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)  # tokenize the pattern
    bag = [0]*len(words)   # bag of words - matrix of N words, vocabulary matrix
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

import tkinter as tk#Library to create Chatbot GUI
from tkinter import *

def send():
    msg = EntryBox.get("1.0",'end-1c').strip() #Get message from textbox and store in msg variable
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n') #In Chat log, insert user message as You:<User Message>
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
    
        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n') #In Chat log, insert Bot response as Bot:<Bot Response>
            
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
 

base = Tk()
base.title("Chat With D!")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

head_label = Label(base, text="Welcome to D's Chat!",width="30",bd=8, bg="white", font="Verdana",anchor="center", fg="blue");
head_label.pack(pady=3)

ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial" ) #Create Chat window
ChatLog.config(state=DISABLED)

scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart") #Add scrollbar to Chat window
ChatLog['yscrollcommand'] = scrollbar.set

SendButton = Button(base, font=("Arial",12,'bold'), text="Send", width="10", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send ) #Create Button to send message

EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial") #Create box to enter message
#EntryBox.bind("<Return>", send)


#Adjust screen layout for all parameters
head_label.place(x=1,y=1,height=40)
scrollbar.place(x=376,y=45, height=400)
ChatLog.place(x=10,y=45, height=400, width=365)
EntryBox.place(x=10, y=450, height=45, width=265)
SendButton.place(x=285, y=450, height=45)

base.mainloop()
