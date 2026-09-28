import argparse
import os
import cv2
import numpy as np
import torch

from dms_utils.dms_utils import load_and_preprocess_image, ACTIONS
from net import MobileNet
from facial_tracking.facialTracking import FacialTracker
import facial_tracking.conf as conf


def infer_one_frame(image, model, yolo_model, facial_tracker):
    eyes_status = ''
    yawn_status = ''
    action = ''

    facial_tracker.process_frame(image)
    if facial_tracker.detected:
        eyes_status = facial_tracker.eyes_status
        yawn_status = facial_tracker.yawn_status

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    yolo_result = yolo_model(rgb_image)

    rgb_image = cv2.resize(rgb_image, (224,224))
    rgb_image = np.expand_dims(rgb_image, 0)
    y = model.predict(rgb_image)
    result = np.argmax(y, axis=1)


    if yolo_result.xyxy[0].shape[0] > 0:
        action = list(ACTIONS.keys())[result[0]]

    cv2.putText(image, f'Driver eyes: {eyes_status}', (30,40), 0, 1,
                conf.LM_COLOR, 2, lineType=cv2.LINE_AA)
    cv2.putText(image, f'Driver mouth: {yawn_status}', (30,80), 0, 1,
                conf.CT_COLOR, 2, lineType=cv2.LINE_AA)
    cv2.putText(image, f'Driver action: {action}', (30,120), 0, 1,
                conf.WARN_COLOR, 2, lineType=cv2.LINE_AA)

    return image


def infer(args):
    image_path = args.image
    video_path = args.video
    cam_id = args.webcam
    checkpoint = args.checkpoint
    save = args.save

    model = MobileNet()
    model.load_weights(checkpoint)

    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
    yolo_model.classes = [67]

    facial_tracker = FacialTracker()

    if image_path:
        image = cv2.imread(image_path)
        image = infer_one_frame(image, model, yolo_model, facial_tracker)
        cv2.imwrite('images/test_inferred.jpg', image)

    if video_path or cam_id is not None:
        cap = cv2.VideoCapture(video_path) if video_path else cv2.VideoCapture(cam_id)

        if cam_id is not None:
            cap.set(3, conf.FRAME_W)
            cap.set(4, conf.FRAME_H)

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if save:
            os.makedirs('videos', exist_ok=True)
            out = cv2.VideoWriter('videos/output.avi',cv2.VideoWriter_fourcc('M','J','P','G'),
                fps, (frame_width,frame_height))

        while True:
            success, image = cap.read()
            if not success:
                break

            image = infer_one_frame(image, model, yolo_model, facial_tracker)
            if save:
                out.write(image)
            else:
                cv2.imshow('DMS', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if save:
            out.release()
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', type=str, default=None, help='Image path')
    p.add_argument('--video', type=str, default=None, help='Video path')
    p.add_argument('--webcam', type=int, default=None, help='Cam ID')
    p.add_argument('--checkpoint', type=str, default='models/model_split.h5', help='Pre-trained model file path')
    p.add_argument('--save', action='store_true', help='Save video instead of displaying it')
    args = p.parse_args()
    if not any((args.image, args.video, args.webcam is not None)):
        p.error('provide --image, --video, or --webcam')
    infer(args)


if __name__ == '__main__':
    main()
