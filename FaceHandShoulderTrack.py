#Import the OpenCV library
import cv2
import numpy as np;
import traceback
from scipy import stats;
import mediapipe as mp


# Load the cascade
face_cascade = cv2.CascadeClassifier('C:/Users/rashm/Desktop/Python/haarcascade_frontalface_default.xml')

#Hands detection
class handDetector():
    def __init__(self,mode = False,maxHands = 2, detectionCon = 0.5,trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode= self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence= self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLM in self.results.multi_hand_landmarks:
                if draw:
                # Draw bounding box around the hand
                    x_min, y_min, x_max, y_max = float('inf'), float('inf'), float('-inf'), float('-inf')
                    for lm in handLM.landmark:
                        x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                        if x < x_min:
                            x_min = x
                        if x > x_max:
                            x_max = x
                        if y < y_min:
                            y_min = y
                        if y > y_max:
                            y_max = y
                    cv2.rectangle(img, (x_min-10, y_min-10), (x_max+10, y_max+10), (0, 255, 0), 2)

        return img
############################################################

#face detection
def detect_face(img_gray):
    # cascade detector to find all faces
    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
    for (x,y,w,h) in faces:
        face = (x,y,w,h);

    # return the face
    return face;

def draw_lines(img, lines, color="GREEN"):
    # generate end points of line
    line_pairs = [(lines[i-1], lines[i]) for i in range(1, len(lines))]
    for line_pair in line_pairs:
        img = draw_line(img, line_pair, color=color)

    return img
# draw line on image
def draw_line(img, line, color="GREEN"):
    (x1,y1) = line[0]
    (x2,y2) = line[1]
    if color == "GREEN":
        color = (0, 255, 0)
    elif color == "BLUE":
        color = (255, 0, 0)
    elif color == "RED":
        color = (0, 0, 255)
    cv2.line(img, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
    return img

def cal_contrast_pixel(img_gray, x, y, h, values=3, search_width=20):
    cols = img_gray[int(y):int(y+h), int(x-search_width/2):int(x+search_width/2)]
    gradient = np.gradient(cols.mean(axis=1), 3)
    max_ind = np.argpartition(np.absolute(gradient), -values)[-values:]
    if np.sum(max_ind) < values:
        return None
    index = np.average(max_ind, weights=np.absolute(gradient[max_ind]))
    return int(y + index)


#shoulder detection
def detect_shoulder(img_gray, x_face, y_face, w_face, h_face, direction, x_scale=0.75, y_scale=0.75):

    # define shoulder box
    w = int(x_scale * w_face)
    h = int(y_scale * h_face)
    y = y_face + h_face * 3/4 # half way down head position
    x = x_face + w_face - w / 20 if direction == "right" else x_face - w + w/20 
    rectangle = (x, y, w, h)

    # calculate position of shoulder
    x_pos, y_pos = [], []
    for delta_x in range(w):
        this_x, this_y = x + delta_x, cal_contrast_pixel(img_gray, x + delta_x, y, h)
        if this_y is not None: 
            x_pos.append(this_x)
            y_pos.append(this_y)

    # extract line from positions
    lines = [(x_pos[index], y_pos[index]) for index in range(len(x_pos))]

    # extract line of best fit from lines
    slope, intercept, _, _, _ = stats.linregress(x_pos, y_pos)
    y0, y1 = int(x_pos[0] * slope + intercept), int(x_pos[-1] * slope + intercept)
    line = [(x_pos[0], y0), (x_pos[-1], y1)]

    # decide on value
    value = np.mean([line[0][1], line[1][1]])

    # return rectangle and positions
    return line, lines, rectangle, value


hist_dict = dict({
    "RIGHT" : [],
    "LEFT" : [],
});
def update_shoulder_history(hist_key, new_value, queue_size=5):
    global hist_dict;
    history = hist_dict[hist_key];
    if(len(history) > queue_size-1): history = history[-queue_size-1:];
    history.append(new_value);
    hist_dict[hist_key] = history;
def detect_shrugging(hist_key, new_value, queue_size = 5, threshold = 6):
    history = (hist_dict[hist_key]);
    history.append(new_value);
    history = np.array(history);
    if(len(history) < queue_size): return False;
    stdev = history.std();
    return stdev > threshold;


def display(capture):
    ret, img = capture.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        face = detect_face(img_gray)
        shoulders = {}
        
        for side in ['right', 'left']:
            line, lines, rectangle, value = detect_shoulder(img_gray, *face, side)
            shrugging = detect_shrugging(side.upper(), value, queue_size=10)
            update_shoulder_history(side.upper(), value, queue_size=10)
            shoulders[side] = {'line': line, 'lines': lines, 'color': 'RED' if shrugging else 'GREEN'}
            
        cv2.rectangle(img, (face[0]-1, face[1]-2), (face[0] + face[2] + 1, face[1] + face[3] + 2), (255, 0, 0), 2)
        for side in shoulders:
            img = draw_lines(img, shoulders[side]['lines'], color='BLUE')
            img = draw_line(img, shoulders[side]['line'], color=shoulders[side]['color'])
        
        detector = handDetector()
        img = detector.findHands(img)
        
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        img = img

    cv2.imshow('video', img)


# Open a video capture object
cap = cv2.VideoCapture('C:/Users/rashm/Downloads/WhatsApp2.mp4')
# Check if the video capture was successfully opened
if not cap.isOpened():
    print("Error: could not open video capture")
else:
    # Loop through frames and display
    while True:
        display(cap)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
