import streamlit as st

from pubmed import run_pubmed

st.title("Pubmed Query and Summary")

query = st.text_input("Enter your pubmed query topic, number of articles requested, and dates if relevant: ")
option = st.radio("Choose summary type", ['General audience', 'Medical audience'])


if st.button("Run"):
    with st.spinner('Running...'):
        initial_state = {'user_input': query}
        
        result = run_pubmed(initial_state)

        st.write("Here's what I found based on your query: ")
        


        if option == 'Medical audience':
            st.write(result.get("summary", "No summary available"))
        if option == 'General audience':
            st.write(result.get("formatted_summary", "No general audience summary available"))