import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.saving import load_model
import re
from PIL import Image
import streamlit as st
import pandas as pd


st.sidebar.title("About Project")
st.sidebar.write("""
This project was developed using Word Embeddings
with 1000 samples and 50 Features with LSTM Neural Network
""")

st.sidebar.title("About Developer")
st.sidebar.write("""
Name: Vijay Kumar\n
Mob:  +919506262955
""")

loaded_model = load_model("review.keras")

file=open("tokens.obj","rb")
tokens=pickle.load(file)
file.close()

def analyze_sentiment(text):
    new_samples=list(map(str.lower,[text]))
    new_samples=list(map(lambda doc:re.sub('[^a-zA-Z ]','',doc),new_samples))

    seqs=tokens.texts_to_sequences(new_samples)	
    pad_seq=pad_sequences(seqs,padding="post",maxlen=32)
    pred=loaded_model.predict(pad_seq)
    return pred.argmax(axis=1)[0]

img=Image.open("banner.png").resize((800,250))
st.image(img)

st.write("Single Prediction")
user_input = st.text_area("Enter Text To Analyze")
btn=st.button("Analyze")

if btn:
    sentiment = analyze_sentiment(user_input)
    if sentiment== 0:
        img=Image.open("dislike.png").resize((200,100))
        st.image(img)
    else:
        img=Image.open("like.png").resize((200,100))
        st.image(img)

st.write("Bulk Prediction")
file=st.file_uploader('Upload Excel File',type=['xlsx'])
if file is not None:
    df=pd.read_excel(file.name,header=None,names=['Review'])
    new_samples=list(map(str.lower,df.Review))
    new_samples=list(map(lambda doc:re.sub('[^a-zA-Z ]','',doc),new_samples))

    seqs=tokens.texts_to_sequences(new_samples)	
    pad_seq=pad_sequences(seqs,padding="post",maxlen=32)
    preds=loaded_model.predict(pad_seq)
    result_df=pd.DataFrame()
    result_df['Review']=df.Review
    result_df['Sentiment']=preds.argmax(axis=1)
    result_df['Sentiment']=result_df['Sentiment'].map({0:"Not Liked",1:"Liked"})
    st.write(result_df)


