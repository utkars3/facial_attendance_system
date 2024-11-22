import streamlit as st


st.set_page_config(page_title='Attendance System',layout='wide')

st.header('Attendance System using Face Recognition')

with st.spinner("Loading Models and Connecting to Database ..."):           #for spinner
    import face_rec

st.success('Model loaded successfully')    
st.success('Redis db successfully connected')