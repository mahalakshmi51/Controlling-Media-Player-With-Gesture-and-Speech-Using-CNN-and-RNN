import cv2
import mediapipe as mp
import pyautogui
import torch.nn as nn
import torch


model=0


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

class CNN_Model(nn.Module):
	def __init__(self, num_channels, num_classes, dropout_rate, dropout_rate_fc):
		super(CNN_Model, self).__init__()
		model.load_state_dict(torch.load("hand_gesture_model", map_location = torch.device('cpu')))
		self.down1 = nn.Sequential(nn.Conv2d(num_channels, 32, kernel_size=3, padding = 1),
		   nn.BatchNorm2d(32),
		   nn.ReLU(),
		   nn.Dropout2d(dropout_rate))
		
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		
		self.down2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding = 1),
		   nn.BatchNorm2d(64),
		   nn.ReLU(),
		   nn.Dropout2d(dropout_rate))
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding = 1)
		
		self.down3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding = 1),
		   nn.BatchNorm2d(128),
		   nn.ReLU(),
		   nn.Dropout2d(dropout_rate))
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
		
		self.ff1 = nn.Sequential(nn.Linear(128 * 7 * 7, 64),  
		nn.BatchNorm1d(64),
		nn.ReLU(),
		nn.Dropout2d(dropout_rate_fc))
		
		self.output = nn.Linear(64, num_classes)
		
	def forward(self, x):
		x = self.down1(x)
		x = self.pool1(x)
		x = self.down2(x)
		x = self.pool2(x)
		x = self.down3(x)
		x = self.pool3(x)
		x = x.reshape(-1, 128 * 7 * 7)
		x = self.ff1(x)
		x = self.output(x)
		return x
     
class RNN_Model(nn.Module):
	def __init__(self, num_channels, num_classes, dropout_rate, dropout_rate_fc):
		super(RNN_Model, self).__init__()
		
		self.down1 = nn.Sequential(nn.Conv2d(num_channels, 32, kernel_size=3, padding = 1),
		   nn.BatchNorm2d(32),
		   nn.ReLU(),
		   nn.Dropout2d(dropout_rate))
		
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		
		self.down2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding = 1),
		   nn.BatchNorm2d(64),
		   nn.ReLU(),
		   nn.Dropout2d(dropout_rate))
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding = 1)
		
		self.down3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding = 1),
		   nn.BatchNorm2d(128),
		   nn.ReLU(),
		   nn.Dropout2d(dropout_rate))
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
		
		self.ff1 = nn.Sequential(nn.Linear(128 * 7 * 7, 64),  
		nn.BatchNorm1d(64),
		nn.ReLU(),
		nn.Dropout2d(dropout_rate_fc))
		
		self.output = nn.Linear(64, num_classes)
        
		
	def forward(self, x):
		x = self.down1(x)
		x = self.pool1(x)
		x = self.down2(x)
		x = self.pool2(x)
		x = self.down3(x)
		x = self.pool3(x)
		x = x.reshape(-1, 128 * 7 * 7)
		x = self.ff1(x)
		x = self.output(x)
		return x


# model.eval()
cap = cv2.VideoCapture(0)
print("Gesture Control Running. Press ESC to exit.")

tip_ids = [4, 8, 12, 16, 20]

def count_fingers(lm_list):
    fingers = []

    # Thumb (check left vs right hand)
    if lm_list[tip_ids[0]].x < lm_list[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other four fingers
    for id in range(1, 5):
        if lm_list[tip_ids[id]].y < lm_list[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = handLms.landmark
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            finger_count = count_fingers(lm_list)
            gesture = None
            # print(finger_count)
            
            if finger_count == 0:
                gesture = "volumemute"
                pyautogui.press('volumemute')
            elif finger_count == 1:
                gesture = "Volume Up"
                pyautogui.press('volumeup')
            elif finger_count == 2:
                gesture = "Volume Down"
                pyautogui.press('volumedown')
            elif finger_count == 3:
                gesture = "Next Track"
                pyautogui.press('Right')
            elif finger_count == 4:
                gesture = "Previous Track"
                pyautogui.press('left')
            elif finger_count == 5:
                gesture = "Play"
                pyautogui.press('playpause')

            if gesture:
                print(f"Detected Gesture: {gesture}")
                cv2.putText(img, f"{gesture}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Gesture Control", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()