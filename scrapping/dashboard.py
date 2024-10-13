import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# Load data
data = pd.read_csv('data_proses.csv')

data['Content'] = data['Content'].apply(lambda x: str(x))

# Menampilkan wordcloud
wordcloud = WordCloud(width=800, height=400, max_words=100).generate(' '.join(data['Content']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

# Menampilkan kata yang sering muncul
text = ' '.join(data['Content'])
words = text.split()
word_counts = Counter(words)
top_words = word_counts.most_common(12)
words, counts = zip(*top_words)
colors = plt.cm.Paired(range(len(words)))
plt.figure(figsize=(10, 6))
bars = plt.bar(words, counts, color=colors)

plt.xlabel('Kata')
plt.ylabel('Frekuensi')
plt.title('Kata yang sering muncul')
plt.xticks(rotation=45)
for bar, num in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2-0.1 ,num + 1, str(num), fontsize=12, color='black', ha='center')
st.pyplot(plt)
