import streamlit as st

from pubmed import run_pubmed

st.title("Pubmed Query and Summary")

query = st.text_input("Enter your pubmed query topic and dates if relevant: ")

if st.button("Run"):
    with st.spinner('Running...'):
        initial_state = {'user_input': query}

        result = run_pubmed(initial_state)
        st.write("Here's what I found based on your query: ")
        st.write(result.get("summary", "No summary available"))

