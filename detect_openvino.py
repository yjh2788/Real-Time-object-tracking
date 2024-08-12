from openvino.tools import mo
from openvino.runtime import serialize
import numpy as np
import torch
from PIL import Image
from utils.datasets import letterbox
from utils.plots import plot_one_box, plot_one_box_PIL
from typing import List, Tuple, Dict
from utils.general import scale_coords, non_max_suppression
from openvino.runtime import Model
from openvino.runtime import Core
import cv2
from utils.torch_utils import time_synchronized
from pathlib import Path
from models.experimental import attempt_load
import face_recognition
import math
from kalmanfilter import KalmanBoxTracker

def face_track(process_this_frame, frame, known_face_encodings, known_face_names):
    if process_this_frame:

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                face_names.append(name)

    process_this_frame = not process_this_frame
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cords = np.array([left, top, right, bottom])
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        if len(face_names):
            return 1, cords
    cords=np.array([0,0,0,0])
    return 0, cords

    # Display the resulting image
    # cv2.imshow('Video', frame)

def face_track_init(user_img="C:/Users/spc/Desktop/user.jpg"):
	my_image = face_recognition.load_image_file(user_img)
	my_face_encoding = face_recognition.face_encodings(my_image)[0]
	return my_face_encoding

def export_onxx(W, H, weight, device='cpu'):
    IMAGE_WIDTH =W  # Suggested values: 2048, 1024 or 512. The minimum width is 512.
    # Set IMAGE_HEIGHT manually for custom input sizes. Minimum height is 512
    IMAGE_HEIGHT = H  # if IMAGE_WIDTH == 2048 else 512
    DIRECTORY_NAME = "model"
    BASE_MODEL_NAME = DIRECTORY_NAME + f"/{weight[:-3]}_{IMAGE_WIDTH}"

    # Paths where PyTorch, ONNX and OpenVINO IR models will be stored
    model_path = Path(BASE_MODEL_NAME).with_suffix(".pth")
    onnx_path = model_path.with_suffix(".onnx")
    ir_path = model_path.with_suffix(".xml")

    # load model
    weights = weight
    model = attempt_load(weights, map_location=device)  # load FP32 model

    # Save the model
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), str(model_path))
    print(f"Model saved at {model_path}")

    if not onnx_path.exists():
        dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)

        # For the Fastseg model, setting do_constant_folding to False is required
        # for PyTorch>1.5.1
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=11,
            do_constant_folding=False,
        )
        print(f"ONNX model exported to {onnx_path}.")
    else:
        print(f"ONNX model {onnx_path} already exists.")

def preprocess_image(img0: np.ndarray,H,W):
    """
    Preprocess image according to YOLOv7 input requirements.
    Takes image in np.array format, resizes it to specific size using letterbox resize, converts color space from BGR (default in OpenCV) to RGB and changes data layout from HWC to CHW.

    Parameters:
      img0 (np.ndarray): image for preprocessing
    Returns:
      img (np.ndarray): image after preprocessing
      img0 (np.ndarray): original image
    """

    # resize
    img = letterbox(img0, auto=False)[0]
    img = cv2.resize(src=img, dsize=(W, H))
    # Convert
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img, img0

def prepare_input_tensor(image: np.ndarray):
    """
    Converts preprocessed image to tensor format according to YOLOv7 input requirements.
    Takes image in np.array format with unit8 data in [0, 255] range and converts it to torch.Tensor object with float data in [0, 1] range

    Parameters:
      image (np.ndarray): image for conversion to tensor
    Returns:
      input_tensor (torch.Tensor): float tensor ready to use for YOLOv7 inference
    """
    input_tensor = image.astype(np.float32)  # uint8 to fp16/32
    input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

    if input_tensor.ndim == 3:
        input_tensor = np.expand_dims(input_tensor, 0)
    return input_tensor

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

def resize_img(image):
    N, C, H, W = model.input(0).shape
    # OpenCV resize expects the destination size as (width, height)
    resized_image = cv2.resize(src=image, dsize=(W, H))
    #print("resized_img shape", resized_image.shape)
    input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0).astype(np.float32)
    #print("img shape=", input_image.shape, "img type=", type(input_image))
    return input_image

def IOU_score(prev,curr):
    p_x1, p_y1, p_x2, p_y2 = prev
    c_x1, c_y1, c_x2, c_y2 = curr

    x_min = max(p_x1, c_x1)
    x_max = min(p_x2, c_x2)
    y_min = max(p_y1, c_y1)
    y_max = min(p_y2, c_y2)

    _AoO = (x_max-x_min)*(y_max-y_min)
    _AoU = (p_x2-p_x1)*(p_y2-p_y1)+(c_x2-c_x1)*(c_y2-c_y1)-_AoO
    score = _AoO/_AoU

    return score

def middle_point_score(prev,curr):
    p_x1, p_y1, p_x2, p_y2 = prev
    c_x1, c_y1, c_x2, c_y2 = curr

    px = p_x2 - p_x1
    py = p_y2 - p_y1
    cx = c_x2 - c_x1
    cy = c_y2 - c_y1

    x_ = abs(cx-px)**2
    y_ = abs(cy-py)**2

    return math.sqrt(x_+ y_)

def square_ratio(prev,curr):
    p_x1, p_y1, p_x2, p_y2 = prev
    c_x1, c_y1, c_x2, c_y2 = curr

    p_width = p_x2 - p_x1
    p_height = p_y2 - p_y1
    c_width = c_x2 - c_x1
    c_height = c_y2 - c_y1

    p_ratio = float(p_height/p_width)
    c_ratio = float(c_height/c_width)
    similarity_ratio = abs(c_ratio-p_ratio)*100
    return similarity_ratio

def check_obj(pre:np.ndarray, curr:np.ndarray):
    pre_boxes=pre[:,:4]
    c_boxes=curr[:,:4]
    score = []
    sim_ratio = []
    for p_box in pre_boxes:
        for c_box in c_boxes:
            score.append(IOU_score(p_box, c_box))
            sim_ratio.append(square_ratio(p_box, c_box))
        min_sim_ratio = max(sim_ratio)
        min_arg = sim_ratio.index(min_sim_ratio)
        #IOU가 0.6이 넘는 애들중 비율이 가장 비슷한 애들
        #max_score = max(score)
        #max_arg = score.index(max_score)
        #if max_score > 0.6 and (min_arg==max_arg):
        #    return max_arg
        th_arg=np.where(np.array(score) >= 0.6)[0]
        #th_arg = score.index(score >=0.6)
        for s in th_arg:
            min_arg = s if sim_ratio[s] < sim_ratio[min_arg] else min_arg
        return min_arg
    return -1

def check_obj_loc(pre:np.ndarray, curr: np.ndarray):
    pre_boxes = pre[:, :4]
    c_boxes = curr[:, :4]
    score = []
    for p_box in pre_boxes:
        for c_box in c_boxes:
            score.append(middle_point_score(p_box, c_box))
        min_score = min(score)
        min_arg = score.index(min_score)
        if min_score > 100:
            return min_arg
    return -1

def detect(video, vid, W, H, model: Model, image_path = 'inference/images/horses.jpg', conf_thres: float = 0.40, iou_thres: float = 0.45, classes: List[int] = None, agnostic_nms: bool = True):
    """
    OpenVINO YOLOv7 model inference function. Reads image, preprocess it, runs model inference and postprocess results using NMS.
    Parameters:
        model (Model): OpenVINO compiled model.
        image_path (Path): input image path.
        conf_thres (float, *optional*, 0.25): minimal accpeted confidence for object filtering
        iou_thres (float, *optional*, 0.45): minimal overlap score for remloving objects duplicates in NMS
        classes (List[int], *optional*, None): labels for prediction filtering, if not provided all predicted labels will be used
        agnostic_nms (bool, *optiona*, False): apply class agnostinc NMS approach or not
    Returns:
       pred (List): list of detections with (n,6) shape, where n - number of detected boxes in format [x1, y1, x2, y2, score, label]
       orig_img (np.ndarray): image before preprocessing, can be used for results visualization
       inpjut_shape (Tuple[int]): shape of model input tensor, can be used for output rescaling
    """

    if video:
        ret, img = vid.read()
    else:
        img = np.array(Image.open(image_path))
        #img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    #input_img=resize_img(img)

    preprocessed_img, orig_img = preprocess_image(img,W,H)
    input_tensor = prepare_input_tensor(preprocessed_img)
    #print("input tensor shape=",input_tensor.shape,"input tensor type=",type(input_tensor))


    output_blob = model.output(0)
    t1 = time_synchronized()
    predictions = torch.from_numpy(model(input_tensor)[output_blob])
    t2 = time_synchronized()
    pred = non_max_suppression(predictions, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    t3 = time_synchronized()
    print(f'({(1E3 * (t2 - t1)):.1f}ms) Inference')
    print(f'({(1E3 * (t3 - t2)):.1f}ms) NMS')


    return pred, orig_img, input_tensor.shape,img

def draw_boxes(p_predictions: np.ndarray, predictions: np.ndarray, input_shape: Tuple[int], image: np.ndarray, names: List[str], colors: Dict[str, int]):
    """
    Utility function for drawing predicted bounding boxes on image
    Parameters:
        predictions (np.ndarray): list of detections with (n,6) shape, where n - number of detected boxes in format [x1, y1, x2, y2, score, label]
        image (np.ndarray): image for boxes visualization
        names (List[str]): list of names for each class in dataset
        colors (Dict[str, int]): mapping between class name and drawing color
    Returns:
        image (np.ndarray): box visualization result
    """
    user = []
    cords = {}
    if not len(predictions):
        #predictions=p_predictions
        return image,cords,p_predictions
    # Rescale boxes from input size to original image size
    predictions[:, :4] = scale_coords(input_shape[2:], predictions[:, :4], image.shape).round()
    ''''''
    s = '%g: '
    # Print results
    ''''''
    arg = check_obj(p_predictions, predictions)
    arg_closest = check_obj_loc(p_predictions, predictions)
    if arg>=0:
        print(predictions)
        predictions[arg,5]=80
        user=predictions[arg].unsqueeze(0)
    elif arg_closest>=0:
        predictions[arg_closest,5]=80
        user=predictions[arg_closest].unsqueeze(0)

    classes={}
    for c in predictions[:, -1].unique():
        n = (predictions[:, -1] == c).sum()  # detections per class
        s += f"{n} {NAMES[int(c)]}{'s' * (n > 1)}, "  # add to string
        classes.setdefault(NAMES[int(c)],int(n))
    print(classes)
    print(s)

    for *xyxy, conf, cls in reversed(predictions):  #상자 좌표, 확율, 클래스
        label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, image, label=label, color=colors[names[int(cls)]], line_thickness=1)
        cords.setdefault(NAMES[int(cls)],xyxy)


    return image, cords, user

def draw_boxes_no(predictions: np.ndarray, input_shape: Tuple[int], image: np.ndarray, names: List[str], colors: Dict[str, int]):
    """
    Utility function for drawing predicted bounding boxes on image
    Parameters:
        predictions (np.ndarray): list of detections with (n,6) shape, where n - number of detected boxes in format [x1, y1, x2, y2, score, label]
        image (np.ndarray): image for boxes visualization
        names (List[str]): list of names for each class in dataset
        colors (Dict[str, int]): mapping between class name and drawing color
    Returns:
        image (np.ndarray): box visualization result
    """

    if not len(predictions):
        return image
    # Rescale boxes from input size to original image size
    predictions[:, :4] = scale_coords(input_shape[2:], predictions[:, :4], image.shape).round()
    ''''''
    s = '%g: '
    # Print results
    ''''''
    for c in predictions[:, -1].unique():
        n = (predictions[:, -1] == c).sum()  # detections per class
        s += f"{n} {NAMES[int(c)]}{'s' * (n > 1)}, "  # add to string
    for *xyxy, conf, cls in reversed(predictions):  #상자 좌표, 확율, 클래스
        label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, image, label=label, color=colors[names[int(cls)]], line_thickness=1)


    return image

''''''
def find_user_box(face_box, detect_box): #아직 모르겟당
    d_boxes = detect_box[:, :4]
    conf = detect_box[:,4]
    cls = detect_box[:,5]
    arg = 0
    for x1, y1, x2, y2 in d_boxes:
        if (face_box[0] > x1) and (face_box[1] > y1) and (face_box[2] < x2) and (face_box[3] < y2):
            return d_boxes[arg].unsqueeze(0)
        else:
            arg += 1
    return np.empty()

if __name__=='__main__':
    size=320
    W=320
    H=320
    weight='yolov7-tiny.pt'
    export_onxx(W,H,weight)
    model = mo.convert_model(f'model/{weight[:-3]}_{size}.onnx')
    # serialize model for saving IR
    serialize(model, f'model/{weight[:-3]}_{size}.xml')
    core = Core()
    # read converted model
    model = core.read_model(f'model/{weight[:-3]}_{size}.xml')
    # load model on CPU device
    compiled_model = core.compile_model(model, 'CPU')

    # video
    vid = cv2.VideoCapture(0)
    #pre_boxes, image, input_shape, frame = detect(1, vid, W, H, compiled_model)
    #p_box = pre_boxes[0]
    #p_box[:, :4] = scale_coords(input_shape[2:], p_box[:, :4], image.shape).round()

    #face-recognition
    my_face_encoding = face_track_init()
    known_face_encodings = [
        my_face_encoding
    ]
    face_locations = []
    known_face_names = [
        "user",
    ]
    # Initialize some variables
    face_encodings = []
    face_names = []
    process_this_frame = True
    #check_user = False ##필요없을 수도
    while True:
        while True:
            ret, frame = vid.read()
            pre_boxes, image, input_shape, frame = detect(1, vid, W, H, compiled_model)
            check_user, cords = face_track(process_this_frame, frame, known_face_encodings, known_face_names)# 얼굴 위치 찾아서 사용자 위치 찾기 구현해야됨
            if check_user:
                p_box = pre_boxes[0]
                p_box[:, :4] = scale_coords(input_shape[2:], p_box[:, :4], image.shape).round()
                user_box = find_user_box(cords,p_box)
                if len(p_box)>0:
                    break

            else:
                boxes, image, input_shape, frame = detect(1, vid, W, H, compiled_model)
                image_with_boxes= draw_boxes_no(boxes[0], input_shape, image, NAMES, COLORS)
                cv2.imshow("img", image_with_boxes)
                cv2.waitKey(1)


        while True:
            boxes, image, input_shape,frame = detect(1,vid,W,H,compiled_model)
            #칼만필터 보정
            kalman_box = KalmanBoxTracker(user_box)
            user_box = torch.from_numpy(kalman_box.predict())
            #
            image_with_boxes, cords ,user_box= draw_boxes(user_box,boxes[0], input_shape, image, NAMES, COLORS)
            if len(user_box) > 0:
                p_box=user_box
            else:
                for i in range(4):
                    boxes, image, input_shape, frame = detect(1, vid, W, H, compiled_model)
                    image_with_boxes, cords, user_box = draw_boxes(p_box, boxes[0], input_shape, image, NAMES, COLORS)
                    if len(user_box) > 0:
                        p_box = user_box
                break

            #face_track(process_this_frame, image_with_boxes,known_face_encodings,known_face_names)


            cv2.imshow("img", image_with_boxes)
            cv2.waitKey(1)


