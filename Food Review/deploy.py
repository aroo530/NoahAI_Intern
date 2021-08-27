import pickle
import streamlit as st
from nltk import word_tokenize
from nltk.corpus import stopwords # Stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer # Stemmer & Lemmatizer

import os
## Clean your reviews using stemmer, lemmatizer & stopwords
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
lemmatizer= WordNetLemmatizer()


def cleaner(sentece):
    word_list=word_tokenize(sentece)
    clean_word_list=[]
    
    for word in word_list:
        if word.lower() not in stop_words:
            lemma=lemmatizer.lemmatize(word)
            stemmed=stemmer.stem(lemma)
            clean_word_list.append(stemmed)
    return " ".join(clean_word_list)


def raw_test(review, model, vectorizer):
    # Clean Review
    review_c = cleaner(review)
    # Embed review using tf-idf vectorizer
    embedding = vectorizer.transform([review_c])
    # Predict using your model
    prediction = model.predict(embedding)
    # Return the Sentiment Prediction
    return "Positive" if prediction == 1 else "Negative"
# this is the main function in which we define our webpage  

def main():
    # front end elements of the web page 
    html_temp = """
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Reviewer ML App</h1> 
    </div> 
    """
    model_name = 'rf_model.pk'
    vectorizer_name = 'tfidf_vectorizer.pk'
    with open(model_name, 'rb') as f:
        loaded_model = pickle.load(f)
    with open(vectorizer_name, 'rb') as f:
        loaded_vect = pickle.load(f)
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
    # following lines create boxes in which user can enter data required to make prediction 
    review = st.text_area('Review',("paste your review here"))
    result =""
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = raw_test(review, loaded_model, loaded_vect) 
        st.success('Your review is {}'.format(result))
        print(result)
if __name__=='__main__': 
    main()