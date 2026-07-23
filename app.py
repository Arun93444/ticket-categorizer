import streamlit as st
import pandas as pd
import joblib
import re
import nltk
import plotly.express as px
from nltk.corpus import stopwords

# ----------------------------
# Download Stopwords
# ----------------------------
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="AI Support Ticket Classifier",
    page_icon="🎫",
    layout="wide"
)

# ----------------------------
# Session State
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("data/support_tickets.csv")

# ----------------------------
# Text Cleaning Function
# ----------------------------
def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^\w\s]", "", text)

    text = re.sub(r"\d+", "", text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🎯 Project Details")

st.sidebar.success("Model : Multinomial Naive Bayes")

st.sidebar.info("Vectorizer : TF-IDF")

st.sidebar.markdown("---")

st.sidebar.subheader("Categories")

st.sidebar.write("💰 Billing")

st.sidebar.write("💻 Technical")

st.sidebar.write("👨‍💼 HR")

st.sidebar.write("📄 General")

st.sidebar.markdown("---")

st.sidebar.subheader("Dataset")

st.sidebar.write(f"Tickets : {len(df)}")

st.sidebar.write(f"Categories : {df['category'].nunique()}")

st.sidebar.write(f"Sources : {df['source'].nunique()}")

# ----------------------------
# Title
# ----------------------------
st.title("🎫 AI Support Ticket Classifier")

st.write(
"""
This application automatically classifies incoming support tickets
using Natural Language Processing (TF-IDF) and Machine Learning.
"""
)

# ----------------------------
# Sample Tickets
# ----------------------------
examples = {

"Billing":
"My payment was deducted twice but refund has not been received.",

"Technical":
"The application crashes every time I login and server is not responding.",

"HR":
"Please send my payslip for last month.",

"General":
"What are your office working hours?"
}

ticket_type = st.selectbox(
    "Choose Sample Ticket",
    list(examples.keys())
)

ticket = st.text_area(
    "Support Ticket",
    examples[ticket_type],
    height=180
)

predict = st.button("🚀 Predict Category")

# ----------------------------
# Prediction
# ----------------------------
if predict:

    # Clean input text
    cleaned_ticket = clean_text(ticket)

    # Convert to TF-IDF vector
    vector = vectorizer.transform([cleaned_ticket])

    # Predict category
    prediction = model.predict(vector)[0]

    # Prediction probabilities
    probabilities = model.predict_proba(vector)[0]

    # Class labels
    classes = model.classes_

    # Confidence
    confidence = round(max(probabilities) * 100, 2)

    # ----------------------------
    # Priority Detection
    # ----------------------------

    urgent_keywords = [
        "urgent",
        "critical",
        "asap",
        "immediately",
        "server down",
        "failed",
        "error",
        "crash",
        "not working",
        "payment failed"
    ]

    priority = "🟢 Normal"

    for word in urgent_keywords:
        if word in ticket.lower():
            priority = "🔴 High"
            break

    # ----------------------------
    # Human Review
    # ----------------------------

    if confidence < 60:
        review = "⚠ Needs Human Review"
    else:
        review = "✅ Auto Assigned"

    st.markdown("---")

    st.subheader("Prediction Result")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Category",
            prediction
        )

    with c2:
        st.metric(
            "Confidence",
            f"{confidence}%"
        )

    with c3:
        st.metric(
            "Priority",
            priority
        )

    st.progress(confidence / 100)

    if confidence < 60:
        st.warning(review)
    else:
        st.success(review)

    # ----------------------------
    # Probability Table
    # ----------------------------

    st.subheader("Prediction Probability")

    prob_df = pd.DataFrame({
        "Category": classes,
        "Probability (%)": (probabilities * 100).round(2)
    })

    st.dataframe(
        prob_df,
        use_container_width=True
    )

    # ----------------------------
    # Probability Chart
    # ----------------------------

    fig = px.bar(
        prob_df,
        x="Category",
        y="Probability (%)",
        text="Probability (%)",
        title="Category Confidence"
    )

    fig.update_traces(textposition="outside")

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ----------------------------
    # Save Prediction History
    # ----------------------------

    st.session_state.history.append({

        "Ticket": ticket,

        "Prediction": prediction,

        "Confidence": confidence,

        "Priority": priority,

        "Status": review

    })
    
        # ----------------------------
    # Prediction History
    # ----------------------------
    st.markdown("---")
    st.subheader("📝 Prediction History")

    history_df = pd.DataFrame(st.session_state.history)

    if not history_df.empty:

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Prediction Report",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )

# ============================
# Dataset Dashboard
# ============================

st.markdown("---")
st.header("📊 Dataset Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Tickets", len(df))

with col2:
    st.metric("Categories", df["category"].nunique())

with col3:
    st.metric("Sources", df["source"].nunique())

# ----------------------------
# Category Distribution
# ----------------------------

category_count = (
    df["category"]
    .value_counts()
    .reset_index()
)

category_count.columns = ["Category", "Count"]

pie = px.pie(
    category_count,
    names="Category",
    values="Count",
    title="Support Ticket Distribution"
)

st.plotly_chart(
    pie,
    use_container_width=True
)

# ----------------------------
# Dataset Preview
# ----------------------------

st.subheader("📋 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True,
    hide_index=True
)

# ============================
# Footer
# ============================

st.markdown("---")

st.markdown(
"""
### 👨‍💻 Developed By

**Arun Prasath**

**AI Support Ticket Classification using TF-IDF & Machine Learning**

**Technologies Used**

- Python
- Streamlit
- Scikit-learn
- TF-IDF Vectorizer
- Multinomial Naive Bayes
- Plotly
- Pandas

"""
)