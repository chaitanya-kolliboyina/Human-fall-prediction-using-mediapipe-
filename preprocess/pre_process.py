from tensorflow.keras.models import load_model
import mediapipe as mp
import os
import pandas as pd
import cv2
import csv
import numpy as np



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# video = cv2.VideoCapture()
    
# width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
# size = (width, height) 
# fps = int(video.get(cv2.CAP_PROP_FPS))

feature_extract = load_model("LSTM_seems_final.h5")

def extract(video):
   

    video = cv2.VideoCapture(video)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height) 
    fps = int(video.get(cv2.CAP_PROP_FPS))

    names = [i.name for  i in mp_pose.PoseLandmark]
    features = []
    for name in names:
        features.append(str(name)+"_x")
        features.append(str(name)+"_y")
        features.append(str(name)+"_z")

    
    # print(csv_path)

    with open("landmarks.csv",'w') as csv_file:
        with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
            csv_out_writer = csv.writer(csv_file, delimiter=',')
            # features.append("Vid_id")
            csv_out_writer.writerow(features)
            while(video.isOpened()):
                success, frame = video.read()
                if success == False:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
                results = pose.process(frame)
   
    
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                landmarks = results.pose_landmarks
                output_frame = frame.copy()
                if landmarks is not None:
                    mp_drawing.draw_landmarks(image = output_frame,landmark_list=landmarks,
                                          connections=mp_pose.POSE_CONNECTIONS)
                output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

                if landmarks is not None:
        # Check the number of landmarks and take pose landmarks.
                    assert len(landmarks.landmark) == 33, 'Unexpected number of predicted pose landmarks: {}'.format(len(landmarks.landmark))
                    landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in landmarks.landmark]
        # print(landmarks)
        # Map pose landmarks from [0, 1] range to absolute coordinates to get
        # correct aspect ratio.

                    frame_height, frame_width = output_frame.shape[:2]
                    landmarks *= np.array([frame_width, frame_height, frame_width])

        # Write pose sample to CSV.
                    landmarks = np.around(landmarks, 5).flatten().astype(int).tolist()
       
        # print(os.path.getsize("trialcsv.csv"))

        
                    csv_out_writer.writerow(landmarks)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
    video.release()
# Closes all the frame
    cv2.destroyAllWindows()
    
    return 'landmarks.csv'

    # features = ["LEFT_SHOULDER_x","LEFT_SHOULDER_y","LEFT_SHOULDER_z","RIGHT_SHOULDER_x","RIGHT_SHOULDER_y","RIGHT_SHOULDER_z","LEFT_ELBOW_x","LEFT_ELBOW_y",
    #           "LEFT_ELBOW_z","RIGHT_ELBOW_x","RIGHT_ELBOW_y","RIGHT_ELBOW_z","LEFT_WRIST_x","LEFT_WRIST_y","LEFT_WRIST_z","RIGHT_WRIST_x","RIGHT_WRIST_y","RIGHT_WRIST_z",
    #           "LEFT_HIP_x","LEFT_HIP_y","LEFT_HIP_z","RIGHT_HIP_x","RIGHT_HIP_y","RIGHT_HIP_z","LEFT_KNEE_x","LEFT_KNEE_y","LEFT_KNEE_z","RIGHT_KNEE_x","RIGHT_KNEE_y",
    #           "RIGHT_KNEE_z","LEFT_ANKLE_x","LEFT_ANKLE_y","LEFT_ANKLE_z","RIGHT_ANKLE_x","RIGHT_ANKLE_y","RIGHT_ANKLE_z","LEFT_HEEL_x","LEFT_HEEL_y","LEFT_HEEL_z",
    #           "RIGHT_HEEL_x","RIGHT_HEEL_y","RIGHT_HEEL_z"]
  
    # d_frame  = pd.read_csv("")

