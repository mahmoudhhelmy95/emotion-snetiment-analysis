import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string
import re
import sklearn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import tkinter
from tkinter import *
#import the data file
dataset=pd.read_excel('C:\\Users\\Mahmoud\\Documents\\Python Scripts\\dataset_final.xlsx')
#filtering_sentence returns clean sentence without punctuation
def filtering_sentence(text):
    stop_words= stopwords.words('arabic')
    filtered_sentence = []
    w = text.translate(str.maketrans("","",string.punctuation))
    splitted_sentence=[]
    splitted_sentence = w.split()
    for word in splitted_sentence:
            if word not in stop_words:
                 filtered_sentence.append(word)
    final_string=str(filtered_sentence)
    return final_string

def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit_transform(train_fit)
    return vector

review_dataset = dataset['review'].apply(filtering_sentence)

tf_vector = get_feature_vector(np.array(review_dataset))
X = tf_vector.transform(np.array(review_dataset))
y = np.array(dataset['label'])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=40)
NB_model = MultinomialNB()
NB_model.fit(X_train, y_train)
y_predict_nb = NB_model.predict(X_test)
print(accuracy_score(y_test, y_predict_nb))


def user_emotion_analysis():
    text1 = entry_1.get()
    final_sentence = filtering_sentence(text1)
    final_sentence = [final_sentence]
    xtxt = get_feature_vector(final_sentence)
    sentence_vector = tf_vector.transform(final_sentence)
    y_predict_nb = NB_model.predict(sentence_vector)
    label2 = Label(root, text=y_predict_nb)
    label2.grid(row=2,column=2)



root = Tk()
root.title("Chatbot")
label1=Label(root,text='enter your review please about out product')
label1.pack()
label1.grid(row=0)
label2=Label(root,text='Feedback')
entry_1=Entry(root)
entry_1.grid(row=1,column=1)
label2.grid(row=1,column=0)


SendButton = Button(root, text="Send", command=user_emotion_analysis)

#SendButton.pack()
SendButton.grid(row=2)
root.mainloop()