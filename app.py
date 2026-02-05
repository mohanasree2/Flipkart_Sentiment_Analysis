import streamlit as st
import joblib

model = joblib.load("flipkart_sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Flipkart Sentiment Analysis")

st.title("🛒 Flipkart Review Sentiment Analysis")
st.write("Enter a product review to predict sentiment")

review = st.text_area("Review Text")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        review_vec = tfidf.transform([review])
        prediction = model.predict(review_vec)[0]

        if prediction == 1:
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😞")



