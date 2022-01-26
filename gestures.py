import cv2
import mediapipe as mp
from openpyxl import Workbook

# Excel
wb = Workbook()
ws = wb.active
row = 0
def get_letter(x):
    return chr(ord('A') + x)

def get_cell(p):
    return f'{p.x},{p.y},{p.z}'

# Training
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
IMAGE_FILES = [] #used to be "hand.png"
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for index, file in enumerate(IMAGE_FILES):
    image = cv2.flip(cv2.imread(file), 1)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
        continue

    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    row = row + 1
    for hand_landmarks in results.multi_hand_landmarks:
        for i in range(0, len(hand_landmarks.landmark)):
            ws[get_letter(i) + str(row)] = get_cell(hand_landmarks.landmark[i])


wb.save("data.xlsx")

# Real-time Classification
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.25,
        min_tracking_confidence=0.25) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        cv2.imwrite("DetectionResults.jpg", image)
        cv2.imshow('MediaPipe Hands', image)

        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            points = results.multi_hand_landmarks

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()