import streamlit as st
import models

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.markdown('Which algo do you wanna use?')
    options = st.selectbox('Which algo are you choosing?',
                           ['MPNET_span', 'SEAv1','ada002'],
                           label_visibility='collapsed')
    
    flag = options
    

#main content
tab1, tab2 = st.tabs(["**Home**","**Results**"])

with tab1:
    st.title('Job Posting here!')
    text = st.text_area('**Please enter the job posting details over here**')
    button_pressed = st.button('Extract Skills')
    if button_pressed:
        extractedskills = models.main_skills_extractor(text, flag)
        
with tab2:
    st.header('Results')
    if button_pressed:
        st.write(extractedskills)  