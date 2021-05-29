from matplotlib.backends.backend_agg import RendererAgg
import streamlit as st
import numpy as np
import pandas as pd
from pandas import json_normalize
import urllib.request
import seaborn as sns
import matplotlib
from matplotlib.figure import Figure
from PIL import Image
import requests
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter

st.set_page_config(layout="wide")

prest_data=pd.read_csv("Prestige amazon.csv")
phil_data=pd.read_csv("philips amazon.csv")
insa_data=pd.read_csv("insala amazon.csv")
hav_data=pd.read_csv("Havells amazon.csv")



matplotlib.use("agg")

_lock = RendererAgg.lock


sns.set_style('darkgrid')
row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.beta_columns(
    (.1, 2, .2, 1, .1))
    
row0_1.title('Analyzing Your Amazon Reviews...')

with row0_2:
    st.write('')

row1_spacer1, row1_1, row1_spacer2 = st.beta_columns((.1, 3.2, .1))

with row1_1:
    st.markdown("Hi there! Let's get on with the analysis of your product")
    st.markdown("**Please upload the File** 👇")

row2_spacer1, row2_1, row2_spacer2 = st.beta_columns((.1, 3.2, .1))
with row2_1:
    uploaded_file= st.file_uploader("Choose a File")
    if uploaded_file is not None:
        pres_data=pd.read_csv(uploaded_file)
    st.markdown("**or**")
    user_input = st.text_input("Input your own Amazon Link")
    
row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.beta_columns(
    (.1, 1, .1, 1, .1))
with row3_1:
    option = st.selectbox(
         'Please select an option',
         ('Data', 'Ratings Distribution','Bigrams/Trigrams', 'Word Clouds','Sentiment of Reviews','Adjectives (POS tagging)'))
    st.write('Displaying the DataFrame/Dashboard you have selected above')
    if option:
        st.write('You selected: ', option)
    

    if option=='Data':
        st.table(pres_data.head())
        
    elif option=='Ratings Distribution':
        st.bar_chart(pres_data["Ratings"].value_counts())
        st.write("The Ratings are as follows:")
        st.text(pres_data["Ratings"].value_counts())
        
    elif option=='Bigrams/Trigrams':
        pres_data['Reviews']=pres_data["Reviews"].apply(lambda x: " ".join(x.lower() for x in x.split()))
        stop = stopwords.words('english')
        pres_data['Reviews'] = pres_data['Reviews'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
        sp = PorterStemmer()
        pres_data['Reviews'] = pres_data['Reviews'].apply(lambda x: " ".join([sp.stem(word) for word in x.split()]))
        c_vec = CountVectorizer(stop_words=stop, ngram_range=(2,3))
           # matrix of ngrams
        ngrams = c_vec.fit_transform(pres_data['Reviews'])
           # count frequency of ngrams
        count_values = ngrams.toarray().sum(axis=0)
           # list of ngrams
        vocab = c_vec.vocabulary_
        df_ngram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
               ).rename(columns={0: 'frequency', 1:'bigram/trigram'})
        df_plot=df_ngram.head(10)
        sns.barplot(x="bigram/trigram", y="frequency", data=df_plot)
        plt.xticks(rotation=45)
        st.pyplot()
        plt.show()
        
      

    elif option=="Sentiment of Reviews":
        sia=SentimentIntensityAnalyzer()
        pres_data['Scores'] = pres_data['Reviews'].apply(lambda text: sia.polarity_scores(text))
        pres_data['Compound']  = pres_data['Scores'].apply(lambda score_dict: score_dict['compound'])
        pres_data['Sentiment'] = pres_data['Compound'].apply(lambda x: 'positive' if x >=0.05 else('negative' if x<= -0.05 else 'neutral'))
        st.bar_chart(pres_data["Sentiment"].value_counts())
        st.text(pres_data["Sentiment"].value_counts())
        st.table(pres_data.head())
    elif option=="Word Clouds":
        def wordcloud(Clean_Content):

            wordcloud_words = " ".join(Clean_Content)
            wordcloud = WordCloud(
                height=300, width=500, background_color="black", random_state=100).generate(wordcloud_words)
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.savefig("cloud.jpg")
            img = Image.open("cloud.jpg")
            return img

        st.header('WordCloud')
        img = wordcloud(pres_data["Reviews"])
        st.image(img)

         

    elif option=='Adjectives (POS tagging)':
        nlp = spacy.load("en_core_web_sm");
        # defining the stop_words and punctuations we want to remove
        punctuations = string.punctuation
        stopwords = spacy.lang.en.stop_words.STOP_WORDS
        words_adj = []
        for line in pres_data['Reviews']:  
          doc= nlp(line) 
          tokens = [tok.lemma_.lower().strip() for tok in doc if tok.pos_ == 'ADJ'] 
          words_adj.append(tokens)
        sia=SentimentIntensityAnalyzer()
        pres_data['Scores'] = pres_data['Reviews'].apply(lambda text: sia.polarity_scores(text))
        pres_data['Compound']  = pres_data['Scores'].apply(lambda score_dict: score_dict['compound'])
        pres_data['Sentiment'] = pres_data['Compound'].apply(lambda x: 'positive' if x >=0.05 else('negative' if x<= -0.05 else 'neutral'))
        pres_data['words_adj'] = pd.Series(words_adj)
   
        positive_adj= pres_data[pres_data["Sentiment"]=='positive']['words_adj']
        pos_words= [line for line in positive_adj for line in set(line)]
        pos_adj_count= Counter(pos_words).most_common(10)
        dfp=pd.DataFrame(pos_adj_count)
        st.table(dfp)
        sns.barplot(x=0, y=1, data=dfp)
        plt.xticks(rotation=45)
        st.pyplot()
        plt.title("Most Common Positive Adjectives");
        plt.show()
        

 
    else:
        st.warning('No option is selected')

