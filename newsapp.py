import streamlit as st

from newsagent import run_news_agent, NewsState

st.title("News Agent")
st.subheader("Ask me anything about recent news and I will give you a summary")

query = st.text_input("Enter your search term here: ")
bias = st.radio("Choose a bias", ['Neutral', 'Right', 'Left'])

if st.button("Run query"):
    with st.spinner('Running query and summarizing...'):
        state = NewsState(user_input = query, user_bias = bias)
        news = run_news_agent(state)
        st.write("Here's what I found based on your query: ")
        st.write(news['final_summary'])
