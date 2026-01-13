import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
# Load Model & Vectorizer
model = pickle.load(open("D:\OASIS INFO BYTES\MAIL\Email-Spam-Detection-main\MNB.pkl", "rb"))
cv = pickle.load(open("D:\OASIS INFO BYTES\MAIL\Email-Spam-Detection-main\cv.pkl", "rb"))
ps = PorterStemmer()
def preprocess_text(message):
    message = re.sub('[^a-zA-Z]', ' ', message)
    message = message.lower()
    message = message.split()
    message = [ps.stem(word) for word in message if word not in stopwords.words('english')]
    return " ".join(message)

st.title("📧 Email Spam Detection App")
st.write("Classify emails as **Spam** or **Not Spam** using Machine Learning.")

email_text = st.text_area("Enter Email Message Below:")

if st.button("Classify Email"):
    if email_text.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned = preprocess_text(email_text)
        vector = cv.transform([cleaned]).toarray()
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.error("🚨 This email is **SPAM**!")
        else:
            st.success("✔️ This email is **NOT SPAM**")
