# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pickle
class Model:
    """docstring for Moddel"""
    def __init__(self):
        self.sc = StandardScaler()
        self.sex_enc=LabelEncoder()		
        self.imputer=Imputer()
        self.classifier = LogisticRegression()
        
    def cleaner(self,text):
        text = text.lower()
        text = re.sub("@[^\s]+","",text)
        text = text.replace(":)","")  
        text = text.replace("@","") 
        text = text.replace("#","") 
        text = text.replace(":(","")
        return text  
    def remove_stop_words(self,text):
        self.sw = stopwords.words("arabic")
        self.clean_words = []
        text = text.split()
        for word in text:
           if word not in self.sw:
               self.clean_words.append(word)
        return " ".join(self.clean_words)
		
    def stemming(self,text):
        self.ps = PorterStemmer()
        text = text.split()
        self.stemmed_words = []
        for word in text :
           self.stemmed_words.append(self.ps.stem(word))
        return " ".join(self.stemmed_words)
    def run(self,text):
        text = self.cleaner(text)
        text = self.remove_stop_words(text)
        text = self.stemming(text)
        return text
    def read_df(self,path):
        self.df = pd.read_csv(path)

    def preprocessing(self):
        self.df['txt']=self.df['txt'].apply(self.run)

    def split_df(self):
        self.tfidf = TfidfVectorizer()
        self.x = self.tfidf.fit_transform(self.df["txt"]).toarray()
        self.y = self.df['sentiment'].values



    def train_test(self,test_size):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size = test_size, random_state = 0)
    
    def train(self,classy):
        self.read_df("ASTD.csv")
        self.preprocessing()
        self.split_df()
        self.train_test(0.25)
        if(classy=="logistic"):
         
         self.classifier.fit(self.x_train,self.y_train)
        if(classy== "SVC"):
           self.classifier = SVC()
           self.classifier.fit(self.x_train, self.y_train)
          
        if(classy=="KNN"):
          self.classifier = KNeighborsClassifier()
          self.classifier.fit(self.x_train, self.y_train)
        self.save_model() 
        self.y_pred=self.classifier.predict(self.x_test)
        return classification_report(self.y_test, self.y_pred)
    def evaluate(self):
        return self.classifier.score(self.x_test,self.y_test)
    def predict(self,test):
        test = self.run(test)
        test = self.tfidf.transform([test]).toarray()
        #test = self.sc.transform([test])
        return self.classifier.predict(test)
        
    def save_model(self):
        save = open("classifier.pickle","wb")
        pickle.dump(self.classifier,save)
        save.close()