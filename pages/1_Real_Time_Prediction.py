import streamlit as st
from Home import face_rec
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import time

# st.set_page_config(page_title='Real Time', layout='centered')
st.subheader('Real Time Attendance System')

#retrive data from redis db
with st.spinner('Retriving Data from Redis DB'):
    redis_face_db=face_rec.retrive_data('academy:enrollment')
    st.dataframe(redis_face_db)

st.success("Data successfully retrived from redis")

#time
waitTime=30 #time in sec for storing the time to redis
setTime=time.time()
realtimepred=face_rec.RealTimePred() #real time face prediction class

#real time prediction

#callback function
def video_frame_callback(frame):
    global setTime

    img = frame.to_ndarray(format="bgr24")  #3d numpy array - bgr24
    print("Processing frame...")
    #operation we can perfomr on array
    pred_img=realtimepred.face_prediction(img,redis_face_db,'facial_features',['Name','Role'],thresh=0.5)

    timenow=time.time()
    difftime=timenow - setTime
    if difftime>=waitTime:
        realtimepred.saveLogs_redis()
        setTime=time.time()

        print('Save data to redis database')

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="realtimeprediction", video_frame_callback=video_frame_callback, rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })