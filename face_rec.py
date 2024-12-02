#this is static file so each time here changes we have to restart server
import numpy as np
import pandas as pd
import cv2

import redis

#insightface
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
#time
import time
from datetime import datetime
import os

#Connect to Redis Client
hostname='redis-15349.c83.us-east-1-2.ec2.redns.redis-cloud.com'
portnumber=15349
password='X41SuRaTQMYfmM0MBfu7WQ8jzNKDDoRr'

r=redis.StrictRedis(host=hostname,
                    port=portnumber,
                    password=password)

#retrive data from database
def retrive_data(name):
    retrive_dict=r.hgetall(name)
    retrive_series=pd.Series(retrive_dict)
    for vector_bytes in retrive_series:
        print(f"Retrieved byte length: {len(vector_bytes)}")
    retrive_series=retrive_series.apply(lambda x:np.frombuffer(x,dtype=np.float32)[:512])
    index=retrive_series.index
    index=list(map(lambda x:x.decode(),index))
    retrive_series.index=index
    retrive_df=retrive_series.to_frame().reset_index() #for dataframe
    retrive_df.columns=['name_role','facial_features']

    retrive_df[['Name','Role']]=retrive_df['name_role'].apply(lambda x:x.split('@')).apply(pd.Series)
    return retrive_df[['Name','Role','facial_features']]

# configure face analysis
faceapp=FaceAnalysis(name='buffalo_sc',root='insightface_model',providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0,det_size=(640,640),det_thresh=0.5)

#ML Search Algo
def ml_search_algorithm(dataframe,feature_column,test_vector,name_role=['Name','Role'],thresh=0.5):
    """
    cosine similarity base search algorithm
    """
    # step-1 Take the dataframe(collection of data)
    dataframe=dataframe.copy()

    # step-2 Index face embedding from the dataframe and convert it into array
    X_list=dataframe[feature_column].tolist()


    x = np.asarray(X_list)
    #X.shape

    # step-3 Calculate cosine similarity
    similar=pairwise.cosine_similarity(x,test_vector.reshape(1,-1))    #1,-1 will also make it to 1X512..1 bata rha hai ek row aur -1 bata rha hai bache hue saare ek hi row me rakho
    similar_arr=np.array(similar).flatten()                            #np.array(similar) kisi bhi tarah ki list ko into a numpy array.. flatten se 1D me convert ho jaiga
    dataframe['cosine']=similar_arr

    # step-4 filter the data
    data_filter=dataframe.query(f'cosine>={thresh}')
    if len(data_filter)> 0 :
        # step-5 get the person name
        data_filter.reset_index(drop=True,inplace=True)
        argmax=data_filter['cosine'].argmax()
        person_name,person_role=data_filter.loc[argmax][name_role]

    else:
        person_name='Unknown'
        person_role='Unknown'

    return person_name,person_role

#real time prediction
#we need to save logs for every 1 min
class RealTimePred:
    def __init__(self):
        self.logs=dict(name=[],role=[],current_time=[])

    def reset_dict(self):    
         self.logs=dict(name=[],role=[],current_time=[])

    def saveLogs_redis(self):
        #step1: create a logs dataframe
        dataframe=pd.DataFrame(self.logs)
        
        #step=2 drop the duplicate information(distinct time) - if a person is detected 10 times no need to save all 
        dataframe.drop_duplicates('name',inplace=True)

        #step-3 push data to redis database (list)  - list only has one value in redis , so we will concate all in one string
        # encode the data
        name_list=dataframe['name'].tolist() 
        role_list=dataframe['role'].tolist() 
        ctime_list=dataframe['current_time'].tolist() 
        encoded_data=[]
         
        for name,role,ctime in zip(name_list,role_list,ctime_list):
            if name !='Unknown':
                concat_string=f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)

        if len(encoded_data)>0:
            r.lpush('attendance:logs',*encoded_data)     

        self.reset_dict()       


    def face_prediction(self,test_image,dataframe,feature_column,name_role=['Name','Role'],thresh=0.5 ):
        #step-0 calculating the time
        current_time=str(datetime.now())


        # step-1 :Take the test image and apply to insight face
        results=faceapp.get(test_image)
        test_copy=test_image.copy()
        
        # step-2 : use for loop and extract each embedding and pass to ml_search_Algo
        for res in results:
            x1,y1,x2,y2=res['bbox'].astype(int)
            embeddings=res['embedding']
            person_name,person_role=ml_search_algorithm(dataframe,feature_column,test_vector=embeddings,name_role=name_role,thresh=thresh)
        
            if person_name == 'Unknown':
                color=(0,0,255) #bgr
        
            else:
                color=(0,255,0)
            cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)
            text_gen=person_name
            #text_gen='hello'
            cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.5,color,1)            #last value is for font size
            cv2.putText(test_copy,current_time,(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX,0.5,color,1)
            #save info in logs dict
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)


        return test_copy
    
    
### Enrollment Form
class EnrollmentFrom:
    def __init__(self):
        self.sample=0

    def reset(self):
        self.sample=0

    def get_embedding(self,frame):
    #get results from insightface model
        results=faceapp.get(frame,max_num=1)   #max_num =1 because we want only 1 person
        embeddings=None    #when there is no face detected it will not go into the loop but we are returning the embeddings therefore we have declared here
        for res in results:
            self.sample+=1
            x1,y1,x2,y2=res['bbox'].astype(int)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)

            #put text sample
            text=f"samples = {self.sample}"
            cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0),2)

            #facial features
            embeddings=res['embedding']
        
        return frame, embeddings
    
    def save_data_in_redis_db(self,name,role):
        #validation name
        if name is not None:
            if name.strip() != '':    #strip will remove all spaces
                key=f'{name}@{role}'
            else:
                return 'name_false'
        else:    
            return 'name_false'

        #if face_embedding.txt exits
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'       

        #step -1 load "face_embedding.txt"
        x_array=np.loadtxt('face_embedding.txt',dtype=np.float32)  #flatten array

        #step -2 convert into array (proper shape)
        received_sample=int(x_array.size/512)  #each face embedding has 512 value
        x_array=x_array.reshape(received_sample,512)
        x_array=np.asarray(x_array) 

        #step-3 cal mean embeddings
        x_mean=x_array.mean(axis=0)  #axis 0 because mean of value of sample not mean of samples
        x_mean=x_mean.astype(np.float32)
        x_mean_bytes=x_mean.tobytes()

        #step-4 save this into redis db
        #redis hashes
        r.hset(name='academy:enrollment',key=key,value=x_mean_bytes)

        #
        os.remove('face_embedding.txt')
        self.reset()

        return True
    


