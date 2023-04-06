from flask import Flask,render_template,redirect,url_for,request,jsonify
import cv2
import os
import numpy as np
import pandas as pd
from preprocess import pre_process
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)
fold = r'uploads'
app.config['UPLOAD_FOLDER'] = fold

clf_path = r"C:\Users\Chaitanya\Documents\Major project\Deploy\Randomforest.sav"
classifier = joblib.load(clf_path)
def process_video(video_file):

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_file.filename))
    video_file.save(video_path)
    # # # Get the uploaded video file
    # video = request.files['video'] 

    landmarks = pre_process.extract(video_path)   
   
    d_frame = pd.read_csv(landmarks)
    d_frame.reset_index(drop=True, inplace=True)
    
    features = ["LEFT_SHOULDER_x","LEFT_SHOULDER_y","LEFT_SHOULDER_z","RIGHT_SHOULDER_x","RIGHT_SHOULDER_y","RIGHT_SHOULDER_z","LEFT_ELBOW_x","LEFT_ELBOW_y",
                   "LEFT_ELBOW_z","RIGHT_ELBOW_x","RIGHT_ELBOW_y","RIGHT_ELBOW_z","LEFT_WRIST_x","LEFT_WRIST_y","LEFT_WRIST_z","RIGHT_WRIST_x","RIGHT_WRIST_y","RIGHT_WRIST_z",
                   "LEFT_HIP_x","LEFT_HIP_y","LEFT_HIP_z","RIGHT_HIP_x","RIGHT_HIP_y","RIGHT_HIP_z","LEFT_KNEE_x","LEFT_KNEE_y","LEFT_KNEE_z","RIGHT_KNEE_x","RIGHT_KNEE_y",
                   "RIGHT_KNEE_z","LEFT_ANKLE_x","LEFT_ANKLE_y","LEFT_ANKLE_z","RIGHT_ANKLE_x","RIGHT_ANKLE_y","RIGHT_ANKLE_z","LEFT_HEEL_x","LEFT_HEEL_y","LEFT_HEEL_z",
                   "RIGHT_HEEL_x","RIGHT_HEEL_y","RIGHT_HEEL_z"]
    
    d_frame = d_frame[features]
    scaler = MinMaxScaler()
    data_rescaled = scaler.fit_transform(d_frame)

    pca = PCA(n_components = 15)
    pca.fit(data_rescaled)
    reduced_99 = pca.transform(data_rescaled)  
    frame_99 = pd.DataFrame(reduced_99)

    frame_99 = frame_99.iloc[304:,:]
    
    
    df_array = np.concatenate([frame_99[col].values.reshape(-1, 1) for col in frame_99.columns], axis=1)
    
    neewww = df_array.reshape(1,df_array.shape[0],df_array.shape[1])
    return neewww

def feature_ext(features_inp):
    # Make a prediction using the machine learning model
    model = load_model("LSTM_seems_final.h5")
    features = model.predict( features_inp)

    # Return the prediction as a JSON object
    return features


@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    if 'file' in request.files:
        # Do your prediction logic here
        # For example, you could get the uploaded file using:
        file = request.files['file']
        
        # And then do some processing on the file to predict the outcome
        inps = process_video(file)
        final = feature_ext(inps)

        # Once you have the predicted result, you can pass it to the result.html template
        result = 'The predicted outcome is...'
        rs = classifier.predict(final)
        if rs[0] ==0:
            pred = "Sit to fall"
        else:
            pred = "Walk to fall"

        print(pred)
        # return render_template('predict.html', prediction=result)
      
    return render_template('predict.html', predict = pred)

if __name__ == '__main__':
    app.run(debug = True)