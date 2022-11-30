from flask import Flask, render_template, Response,request
import face_recognition

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from IPython.display import display
import cv2
from werkzeug.utils import secure_filename
from datetime import datetime

import os

path ='train'

known_face_encodings= []
known_face_names = []
mylist = os.listdir(path)
for cls in mylist:
    curimg = cv2.imread(f'{path}/{cls}')
    
    
    face_1 = curimg
    if(face_recognition.face_encodings(face_1)!=[]):
        face_1_encoding = face_recognition.face_encodings(face_1)[0]
        known_face_encodings.append(face_1_encoding)
        known_face_names.append(os.path.splitext(cls)[0])


 
def makeAttendanceEntry(name):
    with open('attendance_list.csv','r+') as FILE:
        allLines = FILE.readlines()
        attendanceList = []
        for line in allLines:
            entry = line.split(',')
            attendanceList.append(entry[0])
        if name not in attendanceList:
            now = datetime.now()
            dtString = now.strftime('%d/%b/%Y, %H:%M:%S')
            FILE.writelines(f'\n{name},{dtString}')
def video(a):
   
    
    file_name = a
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    pil_image = Image.fromarray(unknown_image)
    draw = ImageDraw.Draw(pil_image)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        # print(known_face_encodings)
        # print(face_encoding)
        print(best_match_index)
    
    if matches[best_match_index]:
        name = known_face_names[best_match_index]
        cv2.rectangle(unknown_image_to_draw,(left, top), (right, bottom), (0,255,0),3 )
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 255))
        cv2.putText(unknown_image_to_draw,name,(left,top-20), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2,cv2.LINE_AA)
        k = ""
        for i in name:
            if(i.isdigit()):
                break
            else:
                k = k+i
        makeAttendanceEntry(k)
        return name
            
                
                
            
                  
    # except:
    #     return "error"
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route("/uploader",methods=['GET','POST'])
def uploader():
    if(request.method == 'POST'):
        f = request.files['file']
        
        
        f.save(os.path.join(secure_filename(f.filename)))
        a = video(f.filename)
        return render_template('done.html')


@app.route("/show",methods=['GET','POST'])
def show():
    csv1 = pd.read_csv('attendance_list.csv')
    
    d = csv1.to_dict('split')
    
    return render_template('show.html',data = d['data'])
