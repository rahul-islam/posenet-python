import tensorflow as tf
import cv2
import time
import argparse
import os
from joblib import dump, load
import posenet
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()

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

def main():

    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.output_dir:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        filenames = [
            f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

        start = time.time()
        for f in filenames:
            input_image, draw_image, output_scale = posenet.read_imgfile(
                f, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.25)
            
            print(input_image.shape[1])
            print(keypoint_coords[0] / [input_image.shape[1], input_image.shape[2]])
            keypoint_coords *= output_scale
            keypoint_coords /= [input_image.shape[1], input_image.shape[2]]
            # keypoint_coords *= 100
            
            clf = load('synthpose.joblib') 

            if args.output_dir:
                draw_image = posenet.draw_skel_and_kp(
                    draw_image, pose_scores, keypoint_scores, keypoint_coords,
                    min_pose_score=0.25, min_part_score=0.25)

                cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), draw_image)

            if not args.notxt:
                print()
                print("Results for image: %s" % f)
                for pi in range(len(pose_scores)):
                    if pose_scores[pi] == 0.:
                        break
                    print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                    t_row = {} #
                    f_df = pd.DataFrame(columns = column_names)
                    for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                        print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
                        if posenet.PART_NAMES[ki] in UNITY_PART_MAP:
                            t_row[UNITY_PART_MAP[posenet.PART_NAMES[ki]] + '_x'] = c[0];
                            t_row[UNITY_PART_MAP[posenet.PART_NAMES[ki]] + '_y'] = c[1];
                    
                    f_df = f_df.append(t_row, ignore_index=True)
                    f_df = f_df.fillna(0)
                    print(f_df)
                    print(clf.predict(f_df))

        print('Average FPS:', len(filenames) / (time.time() - start))

def unitCoords(coords, oldResolution):
    unitCoords = {}
    unitCoords['x'] = coords['x'] / oldResolution['x'];
    unitCoords['y'] = coords['y'] / oldResolution['y']
    return unitCoords;

if __name__ == "__main__":
    main()
