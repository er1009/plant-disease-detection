from firebase import Firebase
import os
# Setting up Firebase configurations
# To learn how to set it up, please go through Firebase_Setup.MD
config = {
    "apiKey": "AIzaSyAQ5ECiTmy1ksMJoogqhHAibVCKbTKUMeg",
    "authDomain": "plant-classifier-4733f.firebaseapp.com",
    "databaseURL": "https://plant-classifier-4733f-default-rtdb.firebaseio.com",
    "projectID": "plant-classifier-4733f",
    "storageBucket": "plant-classifier-4733f.appspot.com",
    "messagingSenderId": "905944574630",
    "appId": "1:905944574630:web:f7464583607a7db83d2267",
    "measurementId": "G-NBB155RHXS"
}

firebase = Firebase(config)

def firebase_gettin():
  '''
  Authentication for Firebase
  '''
  auth = firebase.auth()

  email = "Enter your email here"
  password = "Enter your password"

  try:
    auth.sign_in_with_email_and_password(email, password)
    print("Successfully Signed In")
  except:
    print("Invalid Credentials")

# The path on the cloud storage of Firebase is dependent as per our setup
def send_img(path_local, path_on_cloud = "images/"):
  '''
  Function to push the file into our Firebase storage
  '''

  # Configuration Information
  img_id = path_local.split("/gdrive/MyDrive/Final_Project/repo/final_project/Data-app/images/")
  path_on_cloud = path_on_cloud + img_id[1]
  storage = firebase.storage()
  storage.child(path_on_cloud).put(path_local)
  os.remove(path_local)


def send_feedback(first_name,last_name,email_id,message):
  '''
  Function to push feedback data to firebase
  '''
  db = firebase.database()
  data = {"f_name":first_name,"l_name":last_name,"email":email_id,"msg" : message}
  db.child("feedback").push(data)

def send_prediction(img_id,pred):
  '''
  Function to push feedback data to firebase
  '''
  db = firebase.database()
  data = {"img_id":img_id,"prediction":pred}
  db.child("predictions").push(data)

