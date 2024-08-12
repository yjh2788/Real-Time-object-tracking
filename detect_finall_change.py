from Serial_com import *
from numpy import random
from universal_change import *



# label names for visualization
NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush', 'user']

# colors for visualization
COLORS = {name: [np.random.randint(0, 255) for _ in range(3)]
          for i, name in enumerate(NAMES)}

if __name__=='__main__':
    size=320
    w=640
    H=480
    degree = 90
    vid=cv2.VideoCapture(0)
    #vid.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    #vid.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
   # vid.set(cv2.CAP_PROP_FPS,15)
    #UART=Serial_com("COM5",115200)

    model=loadModel(weights='yolov7-tiny.pt',device='cuda')
    gpu_init()

    #face-recognition
    my_face_encoding = face_track_init()
    known_face_encodings = [
        my_face_encoding
    ]
    face_locations = []
    known_face_names = [
        "user",
    ]

    process_this_frame = True
    #check_user = False ##필요없을 수도
    while True:
        while True:######### 사용자 없으면
            #stop_dynamixel(UART)
            pre_boxes, image, input_shape, frame = detect(1, vid, size, size, model,device='cuda')
            check_user, cords = face_track(process_this_frame, frame, known_face_encodings, known_face_names)# 얼굴 위치 찾아서 사용자 위치 찾기 구현해야됨
            if check_user:# 사용자 찾으면
                p_box = pre_boxes[0]
                p_box[:, :4] = scale_coords(input_shape[2:], p_box[:, :4], image.shape).round()
                p_user_box,flag = find_user_box(cords,p_box)

                if flag != 0:
                    break

            else:######### 사용자 없으면
                #stop_dynamixel(UART)
                boxes, image, input_shape, frame = detect(1, vid, size, size, model,device='cuda')
                image_with_boxes= draw_boxes_no(boxes[0], input_shape, image, NAMES, COLORS)
                cv2.imshow("img", image_with_boxes)
                cv2.waitKey(1)


        while True:########사용자 찾으면
            count=0
            boxes, image, input_shape,frame = detect(1,vid,size,size,model,device='cuda')
            #칼만필터 보정
            non_filtered_user_box=p_user_box#.cpu().numpy()
            kalman_box = KalmanBoxTracker(non_filtered_user_box)
            p_user_box = torch.from_numpy(kalman_box.predict())
            #p_user_box=kalman_box.predict()
            #
            image_with_boxes, cords ,user_box= draw_boxes(p_user_box,boxes[0], input_shape, image, NAMES, COLORS)
            cv2.imshow("img", image_with_boxes)
            cv2.waitKey(1)
            if len(user_box) > 0:
                p_user_box=user_box
                #move_dynamixel(user_box,UART)
                #send_distance(user_box, UART)
            else:

                for i in range(10):
                    count=count+1
                    boxes, image, input_shape, frame = detect(1, vid, size, size, model,device='cuda')
                    image_with_boxes, cords, user_box = draw_boxes(p_user_box, boxes[0], input_shape, image, NAMES, COLORS)
                    cv2.imshow("img", image_with_boxes)
                    cv2.waitKey(1)
                    if len(user_box) > 0:
                        p_user_box = user_box
                        #move_dynamixel(user_box, UART)
                        #send_distance(user_box, UART)
                        break
                if count>=10:
                    #stop_dynamixel(UART)
                    break

            #face_track(process_this_frame, image_with_boxes,known_face_encodings,known_face_names)

            cv2.imshow("img", image_with_boxes)
            cv2.waitKey(1)

