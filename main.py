# -*- coding: utf-8 -*-
import streamlit as st
#from PIL import Image
import cv2 
#import numpy as np
import tempfile

def main():
    st.header("AIClub Face Detection using haarcascade 5")
    uploaded_file=st.file_uploader("Please show us your  photos!!")
    if uploaded_file is not None:
        print("AIClub: At start")
        tfile=tempfile.NamedTemporaryFile(delete=False)
        print("AIClub: Step 10")
        tfile.write(uploaded_file.read())
        image1 = cv2.imread(tfile.name)
        image1b = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
        st.image(image1b, use_column_width=True,clamp = True)
        image2 = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
        print("AIClub: Step 30")
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(image2)
        print(f"AIClub: {len(faces)} faces detected in the image: {faces}.")
        print("AIClub: Step 50")
        for x, y, width, height in faces:
            cv2.rectangle(image2, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=10)
        #cv2.imwrite(tfile.name, image2)
        print("AIClub: Step 70")
        st.image(image2, use_column_width=True,clamp = True)
        print("AIClub: At end")
 
if __name__ == "__main__":
    main()
