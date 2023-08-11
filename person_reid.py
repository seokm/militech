import torchreid
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from typing import List
import numpy as np
from ultralytics import YOLO
import cv2
from sklearn.cluster import KMeans

model_yolo = YOLO("yolov8x.pt")

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))

# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)

# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections, 
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)
    
    tracker_ids = [None] * len(detections)
    
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

CLASS_NAMES_DICT = model_yolo.model.names
CLASS_ID = [0]
byte_tracker = BYTETracker(BYTETrackerArgs())

# settings
LINE_START = Point(50, 1500)
LINE_END = Point(3840-50, 1500)

# Set the resolution of the video

box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
line_counter = LineCounter(start=LINE_START, end=LINE_END)
line_annotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)

# Step 1: Load a pre-trained model
torchreid.data.register_image_dataset('custom', (None, None))
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=0,  # This doesn't matter since we're using a pre-trained model
    loss='softmax',
    pretrained=True
)
model = model.cuda()
weights_path = 'osnet_x1_0_imagenet.pth'
torchreid.utils.load_pretrained_weights(model, weights_path)


# Step 2: Prepare the model for feature extraction
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_dominant_color(image, k=3):
    """Extract dominant colors from an image using KMeans."""
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    
    clf = KMeans(n_clusters=k)
    labels = clf.fit_predict(image)
    
    counts = np.bincount(labels)
    dominant_color = clf.cluster_centers_[np.argmax(counts)]
    
    return dominant_color / 255.0  # Normalize to [0, 1]

def extract_features_from_bbox(img, bbox, visualize = False, maintain_aspect_ratio = False):
    x, y, w, h, no = [int(v) for v in bbox]
    if y + h > 640:
        y_h = 640
    else:
        y_h = y + h
    person_img = img[y:y+h, x:x+w]

    # Assuming upper body is the top 50% of the bounding box
    upper_body = person_img[:h//2, :]
    lower_body = person_img[h//2:, :]

    upper_dominant_color = extract_dominant_color(upper_body)
    lower_dominant_color = extract_dominant_color(lower_body)

    # Convert to RGB as most neural networks expect this format
    person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

    # Visualization for debugging
    if visualize:
        cv2.imshow('Person Image', person_img_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Prepare the image for the model
    person_img_tensor = transform(person_img_rgb).unsqueeze(0).cuda()

   # Extract features from the reID model
    with torch.no_grad():
        appearance_features = model(person_img_tensor).cpu().numpy()

    # Reshape the colors to make them 2D arrays
    upper_dominant_color = upper_dominant_color.reshape(1, -1)
    lower_dominant_color = lower_dominant_color.reshape(1, -1)

    # Combine appearance features and dominant colors
    combined_features = np.concatenate([appearance_features, upper_dominant_color, lower_dominant_color], axis=1)

    return combined_features

print_pictures = []

def get_person_bboxes(frame):
    global print_pictures
    results = model_yolo.predict(frame)
    detections = Detections(
        xyxy=results[0].boxes.xyxy.cpu().numpy(),
        confidence=results[0].boxes.conf.cpu().numpy(),
        class_id=results[0].boxes.cls.cpu().numpy().astype(int)
    )
     # filtering out detections with unwanted classes
    mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
    detections.filter(mask=mask, inplace=True)
    tracks = byte_tracker.update(
        output_results=detections2boxes(detections=detections),
        img_info=frame.shape,
        img_size=frame.shape
    )
    tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
    detections.tracker_id = np.array(tracker_id)
    mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
    detections.filter(mask=mask, inplace=True)
    # format custom labels
    labels = [
        f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections
    ]
    # updating line counter
    line_counter.update(detections=detections)
    frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
    line_annotator.annotate(frame=frame, line_counter=line_counter)
    person_boxes = []
    for idx in range(len(detections.xyxy)):
        bottom_right_x = detections.xyxy[idx][0]
        bottom_right_y = detections.xyxy[idx][1]
        top_left_x = detections.xyxy[idx][2]
        top_left_y = detections.xyxy[idx][3]
        centor_x = (top_left_x + bottom_right_x)/2
        centor_y = (top_left_y + bottom_right_y)/2
        height = top_left_y-bottom_right_y
        person_boxes.append([bottom_right_x,bottom_right_y,top_left_x-bottom_right_x,height,tracker_id[idx]])
    print_pictures.append(frame)
    return person_boxes

# Step 3: Extract features from the images
img1_path = 'img3.jpg'
img2_path = 'img4.jpg'

image1 = cv2.imread(img1_path)
image2 = cv2.imread(img2_path)

image1 = frame = cv2.resize(image1,(640,480))
image2 = frame = cv2.resize(image2,(640,480))

person_bboxes_img1 = get_person_bboxes(image1)
person_bboxes_img2 = get_person_bboxes(image2)

# Here, just for demonstration, I'm comparing the first detected person in each image.
# In a real-world scenario, you might want to compare every person from one image with every person from another image.


matching = []

for i in person_bboxes_img1:
    features1 = extract_features_from_bbox(image1, i)
    features1_tensor = torch.tensor(features1)
    a = []
    b = []
    for j in person_bboxes_img2:
        features2 = extract_features_from_bbox(image2, j)
        features2_tensor = torch.tensor(features2)
        a.append(torch.nn.functional.cosine_similarity(features1_tensor, features2_tensor))
        b.append(j[4])
    idx = a.index(max(a))
    id_num = b[idx]
    if(max(a) > 0.7):
        matching.append([i[4],id_num,max(a)])


for i,j,k in matching:
    print(f"image1's {i} == image2's {j},  similarity == {k}")


# Draw bounding boxes on images to visualize and compare
cv2.imshow('Image 1', print_pictures[0])
cv2.imshow('Image 2', print_pictures[1])
cv2.waitKey(0)
cv2.destroyAllWindows()

