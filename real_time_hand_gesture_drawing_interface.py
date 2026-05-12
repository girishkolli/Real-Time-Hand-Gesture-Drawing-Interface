import os
import cv2
import numpy as np
import mediapipe as mp

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
HEADER_HEIGHT = 100
BRUSH_THICKNESS = 15
ERASER_THICKNESS = 50

COLORS = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 100, 0),
    'eraser': (0, 0, 0)
}


class HandDetector:
    
    def __init__(self, max_hands=1, detection_confidence=0.8, tracking_confidence=0.8):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=1
        )
        self.results = None
        
    def find_hands(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_rgb)
        return frame
    
    def get_finger_positions(self):
        if not self.results or not self.results.multi_hand_landmarks:
            return None, None
        
        hand_landmarks = self.results.multi_hand_landmarks[0]
        h, w = CAMERA_HEIGHT, CAMERA_WIDTH
        
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        
        index_pos = (int(index_tip.x * w), int(index_tip.y * h))
        middle_pos = (int(middle_tip.x * w), int(middle_tip.y * h))
        
        return index_pos, middle_pos
    
    def get_fingers_up(self):
        if not self.results or not self.results.multi_hand_landmarks:
            return [False] * 5
        
        hand_landmarks = self.results.multi_hand_landmarks[0]
        fingers = []
        
        tip_ids = [4, 8, 12, 16, 20]
        
        if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
            fingers.append(True)
        else:
            fingers.append(False)
        
        for id in range(1, 5):
            if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
                fingers.append(True)
            else:
                fingers.append(False)
        
        return fingers


class GestureDrawingApp:
    
    def __init__(self):
        self.cap = None
        self.canvas = None
        self.hand_detector = HandDetector()
        
        self.current_tool = 'red'
        self.draw_color = COLORS['red']
        self.brush_thickness = BRUSH_THICKNESS
        self.prev_x, self.prev_y = None, None
        
        self.zone_width = CAMERA_WIDTH // 4
        
        self.headers = self.load_headers()
        self.current_header = self.headers['red']
        
        self.canvas_gray = None
        self.mask_inv = None
        
    def load_headers(self):
        headers = {}
        header_folder = "selection menu"
        
        for tool in ['red', 'green', 'blue', 'eraser']:
            img_path = os.path.join(header_folder, f"{tool}.png")
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (CAMERA_WIDTH, HEADER_HEIGHT))
                headers[tool] = img
            else:
                print(f"Warning: Could not load {img_path}")
                headers[tool] = np.zeros((HEADER_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        
        return headers
    
    def initialize_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.canvas = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        
        print("Camera initialized successfully")
        
    def get_interaction_mode(self, fingers_up):
        if fingers_up[1] and fingers_up[2]:
            return 'selection'
        
        if fingers_up[1] and not fingers_up[2]:
            return 'drawing'
        
        return None
    
    def select_tool(self, x, y):
        if y >= HEADER_HEIGHT:
            return
        
        zone_index = x // self.zone_width
        tools = ['red', 'green', 'blue', 'eraser']
        
        if 0 <= zone_index < 4:
            self.current_tool = tools[zone_index]
            self.draw_color = COLORS[self.current_tool]
            self.brush_thickness = ERASER_THICKNESS if self.current_tool == 'eraser' else BRUSH_THICKNESS
            self.current_header = self.headers[self.current_tool]
    
    def draw_on_canvas(self, x, y):
        if self.prev_x is None or self.prev_y is None:
            self.prev_x, self.prev_y = x, y
            return
        
        cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), 
                 self.draw_color, self.brush_thickness)
        
        self.prev_x, self.prev_y = x, y
        
        self.canvas_gray = None
    
    def draw_visual_feedback(self, frame, mode, index_pos, middle_pos):
        if index_pos is None:
            return frame
        
        if mode == 'selection' and middle_pos is not None:
            cv2.rectangle(frame, index_pos, middle_pos, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, index_pos, 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, middle_pos, 10, (255, 0, 255), cv2.FILLED)
            
        elif mode == 'drawing':
            cv2.circle(frame, index_pos, 15, self.draw_color, cv2.FILLED)
            cv2.circle(frame, index_pos, 15, (255, 255, 255), 2)
        
        return frame
    
    def overlay_canvas_and_header(self, frame):
        if self.canvas_gray is None:
            self.canvas_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, self.mask_inv = cv2.threshold(self.canvas_gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        frame_bg = cv2.bitwise_and(frame, frame, mask=self.mask_inv)
        canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=cv2.bitwise_not(self.mask_inv))
        
        combined = cv2.add(frame_bg, canvas_fg)
        
        combined[0:HEADER_HEIGHT, 0:CAMERA_WIDTH] = self.current_header
        
        return combined
    
    def run(self):
        self.initialize_camera()
        
        print("Real-time Hand Gesture Drawing Interface Started!")
        print("Controls:")
        print("  - Two fingers up (index + middle): Selection Mode")
        print("  - One finger up (index only): Drawing Mode")
        print("  - Press 'c' to clear canvas")
        print("  - Press 'q' to quit")
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_pos = (10, CAMERA_HEIGHT - 10)
        
        while True:
            success, frame = self.cap.read()
            if not success:
                print("Failed to read from camera")
                break
            
            frame = cv2.flip(frame, 1)
            
            frame = self.hand_detector.find_hands(frame)
            
            index_pos, middle_pos = self.hand_detector.get_finger_positions()
            fingers_up = self.hand_detector.get_fingers_up()
            
            mode = self.get_interaction_mode(fingers_up)
            
            if mode == 'selection' and index_pos is not None:
                self.prev_x, self.prev_y = None, None
                self.select_tool(index_pos[0], index_pos[1])
                    
            elif mode == 'drawing' and index_pos is not None:
                if index_pos[1] > HEADER_HEIGHT:
                    self.draw_on_canvas(index_pos[0], index_pos[1])
                else:
                    self.prev_x, self.prev_y = None, None
            else:
                self.prev_x, self.prev_y = None, None
            
            frame = self.draw_visual_feedback(frame, mode, index_pos, middle_pos)
            
            final_frame = self.overlay_canvas_and_header(frame)
            
            mode_text = f"Mode: {mode.upper() if mode else 'NONE'}"
            tool_display = f"{self.current_tool.upper()} BRUSH" if self.current_tool != 'eraser' else "ERASER"
            tool_text = f"Tool: {tool_display}"
            cv2.putText(final_frame, mode_text, text_pos, font, 0.7, (255, 255, 255), 2)
            cv2.putText(final_frame, tool_text, (10, CAMERA_HEIGHT - 35), font, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Real-time Hand Gesture Drawing Interface", final_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.canvas = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
                self.canvas_gray = None
                print("Canvas cleared")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Real-time Hand Gesture Drawing Interface closed")


def main():
    app = GestureDrawingApp()
    app.run()


if __name__ == "__main__":
    main()