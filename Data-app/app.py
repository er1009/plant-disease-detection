import streamlit as st
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import time
import torch
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import plotly.express as px
from img_classifier import our_image_classifier
import firebase_bro
from text import my_text

# Just making sure we are not bothered by File Encoding warnings
st.set_option('deprecation.showfileUploaderEncoding', False)


def main():
    # Metadata for the web app
    st.set_page_config(
    page_title = "Title of the webpage",
    layout = "centered",
    page_icon= ":shark:",
    initial_sidebar_state = "collapsed",
    )
    menu = ['Home', 'About', 'Contact', 'Feedback','Cannabis plant Diseases']
    choice = st.sidebar.selectbox("Menu", menu)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    labels = {0:'Septoria' ,
         1:'Powdery Mildew',
         2:'Healthy' ,
         3:'Tobacco Mosiac Virus',
         4: 'Spider Mites',
         5:'Calcium Deficiency' ,
         6:'Magnesium Deficiency' }

    treatment, exp = my_text()

    model_plant_detector = torch.load("/gdrive/MyDrive/Final_Project/repo/checkpoints/resnet101_epoch180_f1=0")
    model_plant_detector = model_plant_detector.eval()

    # model = torch.load("/gdrive/MyDrive/Final_Project/repo/checkpoints/vit_b_16_epoch152_f1=80.3035794614742")
    model = torch.load("/gdrive/MyDrive/Final_Project/repo/checkpoints/vit_b_16_epoch76_f1=77.41000066794881")

    if choice == "Home":
        # Let's set the title of our awesome web app
        st.title('Cannabis Disease Doctor')
        # Option to upload an image file with jpg,jpeg or png extensions
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
        # When the user clicks the predict button
        if st.button("Predict"):
        # If the user uploads an image
            if uploaded_file is not None:
                # Opening our image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)
                opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
                image = opencv_image
                local_path = "/gdrive/MyDrive/Final_Project/repo/final_project/Data-app/images/" + uploaded_file.name
                cv2.imwrite(local_path, image)
                # # Send our image to database for later analysis
                firebase_bro.send_img(local_path)
                

                # Let's see what we got
                # st.image(image,use_column_width=True)
                
                st.write("")
                
                try:
                    with st.spinner("The magic of our AI has started...."):
                        label_id, probabilitis, rect_image, HeatMap, grayscale_cam, leaf_detected = our_image_classifier(image, model_plant_detector, model)
                        if leaf_detected:
                          label = labels[label_id]
                          st.success("predicted label is " + label)
                          if not label.lower() == 'healthy':
                            rect_image = cv2.copyMakeBorder(rect_image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, None, value = 0)
                            HeatMap = cv2.copyMakeBorder(HeatMap, 5, 5, 5, 5, cv2.BORDER_CONSTANT, None, value = 0)
                            image_conc = cv2.hconcat([HeatMap, rect_image])
                            st.header("Heat Map & Regions Of Interest")
                            st.image(image_conc,use_column_width=True)
                          
                          firebase_bro.send_prediction(uploaded_file.name, label)
                    if leaf_detected:
                      text_treatment = {
                        0:treatment[0] ,
                        1:treatment[1],
                        2:treatment[2],
                        3:treatment[3],
                        4:treatment[4],                      
                        5:treatment[5] ,
                        6:treatment[6] }
                     
                      na = probabilitis.detach().cpu().numpy().tolist()[0]
                      # st.success(na)
                      dict_ = {"prob":na, "label":["Septoria", "Powdery Mildew", " healthy","TMV",
                                "Spider mites","Calcium deficiencies","Magnesium deficiency"]}
                      df = pd.DataFrame(data=dict_)

                      # fig = px.bar(x, y)
                      st.header("Classes Probability Gragh")
                      fig = px.bar(df, x = 'label', y = 'prob')
                      st.plotly_chart(fig, use_container_width=True)
                      if not label.lower() == 'healthy':
                        st.header("Disease Treatment")
                      st.text(text_treatment[label_id])
                  
                      #rating = st.slider("Do you mind rating our service?",1,10)
                except:
                    st.error("We apologize something went wrong 🙇🏽‍♂️")
            else:
                st.error("Can you please upload an image 🙇🏽‍♂️")

    elif choice == "Contact":
        # Let's set the title of our Contact Page
        # st.title('Get in touch')
        def display_team(name,path,affiliation="",email1="", email2="",email3=""):
            '''
            Function to display picture,name,affiliation and name of creators
            '''
            team_img = Image.open(path)

            st.image(team_img, width=350, use_column_width=False)
            st.markdown(f"## {name}")
            # original_title = '<p style="font-family:Courier; color:Blue; font-size: 20px;">HERE I AM</p>'
            # st.markdown(original_title, unsafe_allow_html=True)
            st.markdown(f"#### {affiliation}")
            st.markdown(f"###### Eldad's Email : {email1}")
            st.markdown(f"###### Jacob's Email : {email2}")
            st.markdown(f"###### Bar's Email : {email3}")
            st.write("------")

        display_team("Team members ", "/gdrive/MyDrive/Final_Project/repo/final_project/Data-app/assets/contact.jfif",
        "Eldad Ron , Jacob Monayer and Bar Yaakobi","Eldad.Ron@s.afeka.ac.il","Jacob.Monayer@s.afeka.ac.il"," Bar.Yaakobi@s.afeka.ac.il")

    elif choice == "About":
        # Let's set the title of our About page
        # st.title('About')

        # A function to display the company logo
        def display_logo(path):
            company_logo = Image.open(path)
            st.image(company_logo, width=350, use_column_width=False)

        # Add the necessary info
        display_logo("./assets/about.jfif")
        st.write("------")
        st.markdown(''' 
        ###### Our system uses models based on algorithms from the field of computer vision to predict the cannabis plant's health condition
        ''')
        st.markdown('###### Main objectives : ')
        st.markdown('''
        ###### 1. Monitoring the plant's health, and diagnose health condition of the plant.
      ''')
        st.markdown(''' 
        ###### 2. Offer the right treatment for each disease, recommended treatments are shown after the diagnosis on the main page.
        ''')
        st.markdown(''' 
        ###### 3. Localize disease findings by visualization of the infected parts on the plant. As growers, This visualization can help to localize the cause of the disease and prevent the plant condition to get worse.
        ''')
        st.markdown('''
        ###### Our services address the needs of the local and global growers community issues, taking care of plant's health and growing process to maximize cannabis yields.
        ''')
        # st.markdown('## the moons ')
        # st.markdown("Write more about your country here.")

    elif choice == "Feedback":
        # Let's set the feedback page complete with a form
        st.title("Feel free to share your opinions :smile:")

        first_name = st.text_input('First Name:')
        last_name = st.text_input('Last Name:')
        user_email = st.text_input('Enter Email: ')
        feedback = st.text_area('Feedback')

        # When User clicks the send feedback button
        if st.button('Send Feedback'):
            # # Let's send the data to a Database to store it
            firebase_bro.send_feedback(first_name, last_name, user_email, feedback)

            # Share a Successful Completion Message
            st.success("Your feedback has been shared!")
    elif choice == "Cannabis plant Diseases":
      st.title(" Diseases Explanation  ")
      st.title("                       ")
      def display_disease(name,path,explanation=""):
            '''
            Function to display picture,name of disease and explanation about the disease
            '''
            disease_image = Image.open(path)
            st.markdown(f"## {name}")
            st.image(disease_image, width=350, use_column_width=False)
            st.text(explanation)
            st.write("------")

      path ="/gdrive/MyDrive/Final_Project/Project_Code/image_copy4/Septoria/S1.jpg"
      display_disease("Septoria",path,exp[0])

      path = "/gdrive/MyDrive/Final_Project/Project_Code/image_copy4/Powdery Mildew/PM7.jpg"
      display_disease("Powdery Mildew",path,exp[1])

      path ="/gdrive/MyDrive/Final_Project/Project_Code/image_copy4/Tobacco Mosiac Virus/TMV12_1.jpg"
      display_disease("Tobacco Mosiac Virus",path,exp[3])

      path = "/gdrive/MyDrive/Final_Project/Project_Code/image_copy4/Spider Mites/SM4.jpg"
      display_disease("Spider Mites",path,exp[4])

      path = "/gdrive/MyDrive/Final_Project/Project_Code/image_copy4/Calcium Deficiency/CD2.jpg"
      display_disease("Calcium Deficiency",path,exp[5])

      path = "/gdrive/MyDrive/Final_Project/Project_Code/image_copy4/Magnesium Deficiency/MD21_1.jpg"
      display_disease("Magnesium Deficiency",path,exp[6])

if __name__ == "__main__":
    main()
