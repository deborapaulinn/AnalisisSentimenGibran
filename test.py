import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import re
#from wordcloud import WordCloud, STOPWORDS
import numpy as np
import Sastrawi #Menghapus kata kata penghubung
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory #Mengubah kata menjadi bahasa asli
from wordcloud import WordCloud, STOPWORDS
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
from textblob import TextBlob
import cleantext
import torch
from torchvision import datasets, transforms

more_stop_words = ["tidak"]

stop_words = StopWordRemoverFactory().get_stop_words()
stop_words.extend(more_stop_words)

new_array = ArrayDictionary(stop_words)
stop_words_remover_new = StopWordRemover(new_array)

def stopword(str_text):
    str_text = stop_words_remover_new.remove(str_text)
    return str_text


def clean_twitter_text(text):
  """Membersihkan teks tweet, termasuk menghapus karakter khusus dan spasi berlebihan."""

  # Menghapus karakter khusus
  text = re.sub(r'@[A-Za-z0-9_]+', '', text)
  text = re.sub(r'#\w+', '', text)
  text = re.sub(r'RT[\s]+', '', text)
  text = re.sub(r'https?://\S+', '', text)

  # Menghapus spasi berlebihan di antara kata-kata
  text = re.sub(r'\s+', ' ', text)

  # Menghapus spasi di awal dan akhir teks
  text = text.strip()

  return text

norm = {' gugel ': ' google ',
        ' buku ':' book ',
        ' kucing ':' cat ',
        # Tambahkan penyesuaian lain sesuai kebutuhan Anda
       }

def normalisasi(str_text):
    for i in norm:
        str_text = str_text.replace(i, norm[i])
    return str_text

#title
st.title('ANALISIS SENTIMEN TERHADAP GIBRAN RAKABUMING DI TWITTER')
#markdown
st.markdown('Ini adalah hasil analisa Kelompok 13 terhadap dataset yang diambil dari Twitter mengenai tweet Gibran.')

#sidebar
st.sidebar.title('Analisis Sentimen Terhadap Gibran')
# sidebar markdown 
st.sidebar.markdown("NLP")
    #loading the data (the csv file is in the same folder)
    #subheader
st.sidebar.subheader('Kelompok 13')
st.sidebar.write('Paulina Agusia - 535220048')
st.sidebar.write('Vincent Calista - 535220075')
st.sidebar.write('Merry Manurung - 535220263')

#if the file is stored the copy the path and paste in read_csv method.
raw = pd.read_csv('.\data_gibran.csv', index_col=0)
df = raw[['full_text', 'username', 'created_at']]
df.drop_duplicates(subset=['full_text'])
df['full_text'] = df['full_text'].apply(clean_twitter_text).str.lower()
df['full_text'] = df['full_text'].apply(lambda X: normalisasi(X))
df['full_text'] = df['full_text'].apply(lambda x: stopword(x))

#checkbox to show data 
if st.checkbox("Show Raw Data"):
    st.write(raw.head(50))

if st.checkbox("Show Cleaned Data"):
    st.write(df.head(50))


tokenized = df['full_text'].apply(lambda x:x.split())

if st.checkbox("Show Tokenized Data"):
    st.write(tokenized.head(50))


all_words = ' '.join([tweet for tweet in df['full_text']])

from transformers import pipeline
#
st.subheader('Sentiment Analysis')
with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        sentiment_pipeline = pipeline("sentiment-analysis", model="indobenchmark/indobert-base-p1")
        result = sentiment_pipeline(text)
        st.write('Polarity:', round(result[0]['score'], 2))

        label = result[0]['label']
        if (label == "LABEL_3"):
            sentiment = "Negative"
        elif (label == "LABEL_0"):
            sentiment = "Positive"
        else:
            sentiment = label

        st.write('Sentiment:', sentiment)

data = pd.DataFrame({'data_gibran': ['positive', 'negative', 'positive', 'neutral', 'positive', 'negative']})
sentiment=data['data_gibran'].value_counts()
sentiment=pd.DataFrame({'Sentiment':sentiment.index,'Tweets':sentiment.values})

st.subheader('Sentiment Analysis')
fig = px.bar(sentiment, x='Sentiment', y='Tweets', color = 'Tweets', height= 500)
st.plotly_chart(fig)

wordcloud = WordCloud(
    width=3000,
    height=2000,
    random_state=3,
    background_color='black',
    colormap='RdPu',
    collocations=False,
    stopwords=STOPWORDS,
).generate(all_words)

# Generate word frequencies
word_freq = wordcloud.words_

# Convert word frequencies to DataFrame
word_freq_df = pd.DataFrame(list(word_freq.items()), columns=['Kata', 'Frekuensi'])

# Sort DataFrame by frequency in descending order
word_freq_df = word_freq_df.sort_values(by='Frekuensi', ascending=False)

# Plot word cloud
st.subheader("Word Cloud")
fig_wc, ax_wc = plt.subplots()
ax_wc.imshow(wordcloud, interpolation='bilinear')
st.pyplot(fig_wc)

# Plot Bar Chart
#st.subheader("Word Cloud")
#fig_wc, ax_wc = plt.subplots()
#ax_wc.imshow(wordcloud, interpolation='bilinear')
#st.pyplot(fig_wc)



#selectbox + visualisation
# An optional string to use as the unique key for the widget. If this is omitted, a key will be generated for the widget based on its content.
## Multiple widgets of the same type may not share the same key.

