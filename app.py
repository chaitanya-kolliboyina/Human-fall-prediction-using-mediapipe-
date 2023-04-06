from flask import Flask, render_template, request, jsonify
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

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
#To extract points from video files

# Load the pre-trained model

# classifier = load_model("C:\Users\Chaitanya\Documents\Major project\Deploy\Randomforest.sav")

# def process_video(video_file):

#     video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_file.filename))
#     video_file.save(video_path)
#     # Get the uploaded video file
#     file = request.files['video']

#     landmarks = pre_process.extract(file)
    

#     d_frame = pd.read_csv(landmarks)
#     d_frame.reset_index(drop=True, inplace=True)

#     features = ["LEFT_SHOULDER_x","LEFT_SHOULDER_y","LEFT_SHOULDER_z","RIGHT_SHOULDER_x","RIGHT_SHOULDER_y","RIGHT_SHOULDER_z","LEFT_ELBOW_x","LEFT_ELBOW_y",
#                    "LEFT_ELBOW_z","RIGHT_ELBOW_x","RIGHT_ELBOW_y","RIGHT_ELBOW_z","LEFT_WRIST_x","LEFT_WRIST_y","LEFT_WRIST_z","RIGHT_WRIST_x","RIGHT_WRIST_y","RIGHT_WRIST_z",
#                    "LEFT_HIP_x","LEFT_HIP_y","LEFT_HIP_z","RIGHT_HIP_x","RIGHT_HIP_y","RIGHT_HIP_z","LEFT_KNEE_x","LEFT_KNEE_y","LEFT_KNEE_z","RIGHT_KNEE_x","RIGHT_KNEE_y",
#                    "RIGHT_KNEE_z","LEFT_ANKLE_x","LEFT_ANKLE_y","LEFT_ANKLE_z","RIGHT_ANKLE_x","RIGHT_ANKLE_y","RIGHT_ANKLE_z","LEFT_HEEL_x","LEFT_HEEL_y","LEFT_HEEL_z",
#                    "RIGHT_HEEL_x","RIGHT_HEEL_y","RIGHT_HEEL_z"]
    
#     d_frame = d_frame[features]

#     scaler = MinMaxScaler()
#     data_rescaled = scaler.fit_transform(d_frame)

#     pca = PCA(n_components = 15)
#     pca.fit(data_rescaled)
#     reduced_99 = pca.transform(data_rescaled)  
#     frame_99 = pd.DataFrame(reduced_99)

#     frame_99 = frame_99.iloc[304:,:]
    
    
#     df_array = np.concatenate([frame_99[col].values.reshape(-1, 1) for col in frame_99.columns], axis=1)
    
#     neewww = df_array.reshape(1,df_array.shape[0],df_array.shape[1])
#     return neewww

# def predict(features):
#     # Make a prediction using the machine learning model
#     model = load_model("Randomforest.sav")
#     prediction = model.predict( features)

#     # Return the prediction as a JSON object
#     return jsonify({'prediction': prediction.tolist()})


@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def upload():
#     if request.method == 'POST':
#         # Get the uploaded file
#         video_file = request.files['file']

#         # Process the video file and make a prediction
#         landmarks = process_video(video_file)
#         prediction = predict(landmarks)

#         return  render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run()
