import tensorflow as tf
import cv2
import time
import argparse

import posenet

from joblib import dump, load
import pandas as pd
column_names = ['Eye_L_x', 'Eye_L_y', 'Eye_R_x', 'Eye_R_y', 'Hip_L_x', 'Hip_L_y',
       'Knee_L_x', 'Knee_L_y', 'Ankle_L_x', 'Ankle_L_y', 'Toes_L_x',
       'Toes_L_y', 'ToesEnd_L_x', 'ToesEnd_L_y', 'Shoulder_L_x',
       'Shoulder_L_y', 'Elbow_L_x', 'Elbow_L_y', 'Wrist_L_x', 'Wrist_L_y',
       'Hip_R_x', 'Hip_R_y', 'Knee_R_x', 'Knee_R_y', 'Ankle_R_x', 'Ankle_R_y',
       'Shoulder_R_x', 'Shoulder_R_y', 'Elbow_R_x', 'Elbow_R_y', 'Wrist_R_x',
       'Wrist_R_y']

UNITY_PART_MAP = {
    # 'nose' : '',
    'leftEye' : 'Eye_L',
    'rightEye' : 'Eye_R',
    # 'leftEar' : '',
    # 'rightEar' : '',
    'leftShoulder' : 'Shoulder_L',
    'rightShoulder' : 'Shoulder_R',
    'leftElbow' : 'Elbow_L',
    'rightElbow' : 'Elbow_R',
    'leftWrist' : 'Wrist_L',
    'rightWrist' : 'Wrist_R',
    'leftHip' : 'Hip_L',
    'rightHip' : 'Hip_R',
    'leftKnee' : 'Knee_L',
    'rightKnee' : 'Knee_R',
    'leftAnkle' : 'Ankle_L',
    'rightAnkle' : 'Ankle_R',
}

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
parser.add_argument('--notxt', action='store_true')
parser.add_argument("-s", "--size", type=int, default=5, help="size of queue for averaging")
args = parser.parse_args()

def unitCoords(coords, oldResolution):
    unitCoords = {}
    unitCoords['x'] = coords['x'] / oldResolution['x'];
    unitCoords['y'] = coords['y'] / oldResolution['y']
    return unitCoords;

def addText(image, text):
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # org 
    org = (50, 50) 

    # fontScale 
    fontScale = 1

    # Blue color in BGR 
    color = (255, 0, 0) 

    # Line thickness of 2 px 
    thickness = 2

    # Using cv2.putText() method 
    image = cv2.putText(image, text, org, font,  
                    fontScale, color, thickness, cv2.LINE_AA) 

    return image


from collections import deque, Counter
Q = deque(maxlen=args.size)

def main():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        cap = cv2.VideoCapture(args.cam_id) # default value
        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)
            
            keypoint_coords *= output_scale
            cp_keypoint_coords = keypoint_coords # copy 
            keypoint_coords /= [input_image.shape[1], input_image.shape[2]]
            # keypoint_coords *= 400

            clf = load('synthpose.joblib')


            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, cp_keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            if not args.notxt:
                for pi in range(len(pose_scores)):
                    if pose_scores[pi] == 0.:
                        break
                    # print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                    t_row = {} #
                    f_df = pd.DataFrame(columns = column_names)
                    for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                        # print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
                        if posenet.PART_NAMES[ki] in UNITY_PART_MAP:
                            t_row[UNITY_PART_MAP[posenet.PART_NAMES[ki]] + '_x'] = c[1];
                            t_row[UNITY_PART_MAP[posenet.PART_NAMES[ki]] + '_y'] = c[0];
                    
                    f_df = f_df.append(t_row, ignore_index=True)
                    f_df = f_df.fillna(0)
                    
                    y = clf.predict(f_df)[0]
                    # print(y, pose_scores[pi])
                    if pose_scores[pi] > 0.4:
                        Q.append(y)
                        b = Counter(Q).most_common(1)[0][0]
                        print (b)
                        overlay_image = addText(overlay_image, b)

            
            cv2.imshow('posenet', overlay_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()