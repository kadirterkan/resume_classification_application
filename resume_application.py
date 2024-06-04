import re
import string
import torch
import json

import nltk
import spacy
import streamlit as st
import torch.nn.functional as F
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from pypdf import PdfReader
from transformers import  AutoTokenizer, DistilBertForSequenceClassification

nlp = spacy.load('en_core_web_sm')
nltk.download('stopwords')

bert_model = DistilBertForSequenceClassification.from_pretrained('./notebooks/bert_model')

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

with open('./notebooks/labels.json', 'r') as f:
    label_map = json.load(f)

def clean_string(text, useless_words=None, stem="None"):
    if useless_words is None:
        useless_words = []
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', '  ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub('\s+', ' ', text)

    text = text.lower()

    text = re.sub(r'\n', '', text)

    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    text = re.sub(r'[0-9]+', '', text)

    text = text.split()
    useless_words = nltk.corpus.stopwords.words("english") + useless_words
    useless_words = useless_words + ['hi', 'im']

    text_filtered = [word for word in text if not word in useless_words]

    text_filtered = [re.sub(r'\w*\d\w*', '', w) for w in text_filtered]

    if stem == 'Stem':
        stemmer = PorterStemmer()
        text_stemmed = [stemmer.stem(y) for y in text_filtered]
    elif stem == 'Lem':
        lem = WordNetLemmatizer()
        text_stemmed = [lem.lemmatize(y) for y in text_filtered]
    elif stem == 'Spacy':
        text_filtered = nlp(' '.join(text_filtered))
        text_stemmed = [y.lemma_ for y in text_filtered]
    else:
        text_stemmed = text_filtered

    final_string = ' '.join(text_stemmed)

    return final_string


def tokenize(text):
    return tokenizer.fit_on_texts(text, truncation=True, padding=True)


def main():
    st.title("Resume Classification App")

    uploaded_file = st.file_uploader("Upload Resume as PDF", type="pdf")
    if uploaded_file is not None:
        reader = PdfReader(uploaded_file)

        text = ""

        for page in range(len(reader.pages)):
            pageObj = reader.pages[0]
            text = text + clean_string(pageObj.extract_text())

        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = bert_model(**inputs)

        logits = outputs.logits

        probabilities = F.softmax(logits, dim=1)

        # Get the top 5 predictions
        top5_prob, top5_label_indices = torch.topk(probabilities, 5)

        # Convert the probabilities and labels to a list
        top5_prob = top5_prob.squeeze().tolist()
        top5_label_indices = top5_label_indices.squeeze().tolist()

        # Map the indices to labels
        top5_labels = [label_map[idx] for idx in top5_label_indices]

        # Combine the labels and their probabilities
        top5_predictions = list(zip(top5_labels, top5_prob))

        st.subheader("Top 5 Predictions")
        st.write(top5_predictions)


if __name__ == "__main__":
    main()
