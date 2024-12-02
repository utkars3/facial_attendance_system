import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av
from Home import face_rec

# st.set_page_config(page_title='Registration',layout='centered')
st.subheader('Registration Form')


## init enrollment
enrollment_form=face_rec.EnrollmentFrom()

#step-1 Collect person name and role
#form
person_name=st.text_input(label='Name',placeholder='Name')
role=st.selectbox(label='Select your role',options=('Student','Teacher'))

#step-2 Collect Facial embeddings of that person
#this is a callback function.we cannot do anykind of appending in this
def video_callback_func(frame):
    img=frame.to_ndarray(format='bgr24') #3d array - bgr
    enroll_img,embedding=enrollment_form.get_embedding(img)

    #two step process
    #1st save save data into local computer txt
    if embedding is not None:
        with open('face_embedding.txt',mode='ab') as f:     #ab- append in bytes
            np.savetxt(f,embedding)

    return av.VideoFrame.from_ndarray(enroll_img,format='bgr24')

webrtc_streamer(key='enrollment',video_frame_callback=video_callback_func, rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

#step-3 Save the data in redis database


if st.button('Submit'):
    return_val=enrollment_form.save_data_in_redis_db(person_name,role)
    if return_val==True:
        st.success(f"{person_name} enrolled successfully")

    elif return_val == 'name_false':
        st.error('Please enter the name : Name cannot be empty or spaces')

    elif return_val == 'file_false':
        st.error('Embedding file is missing.Please refresh the page and execute again')



