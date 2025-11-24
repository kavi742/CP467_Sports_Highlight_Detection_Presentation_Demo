import cv2
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv
import os
from datetime import datetime
import argparse
from inference_sdk import InferenceHTTPClient

class SoccerAnalysisSystem:
    def __init__(self, output_base_name, roboflow_api_key):
        # Initialize device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load YOLO model for player and ball detection
        self.model = YOLO('yolov8s.pt')
        self.model.to(self.device)
        
        # Initialize tracker for ball
        self.tracker = sv.ByteTrack()
        
        # Initialize Roboflow client for goalpost detection
        self.roboflow_client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=roboflow_api_key
        )
        self.goalpost_model_id = "football-goalpost/3"
        
        # Goal detection state
        self.last_homography = None
        self.last_goal_corners = None
        self.goal_detected = False
        self.goal_scored = False
        self.goal_frame_count = 0
        self.consecutive_goal_detections = 0
        self.min_consecutive_detections = 3
        
        # Hough Transform parameters for goal structure refinement
        self.hough_params = {
            'rho': 1,
            'theta': np.pi / 180,
            'threshold': 50,
            'min_line_length': 40,
            'max_line_gap': 20
        }
        
        # Output configuration
        self.output_base_name = output_base_name
        self.output_video_path = f"OUTPUT_{output_base_name}.mp4"
        self.goal_images_dir = f"OUTPUT_{output_base_name}_goal_moments"
        
        # Sports ball class ID in COCO dataset (YOLO)
        self.BALL_CLASS_ID = 32
        self.PERSON_CLASS_ID = 0
        
        # Colors for visualization
        self.colors = {
            'goalpost_ml': (0, 255, 0),      # Green for ML-detected goalposts
            'goalpost_hough': (255, 255, 0), # Yellow for Hough lines
            'goal_area': (0, 255, 255),      # Cyan for goal area
            'ball': (255, 0, 255),           # Pink for ball
            'player': (255, 0, 0),           # Blue for players
            'goal_text': (0, 255, 255),      # Yellow for goal text
            'goal_moment': (0, 165, 255),    # Orange for goal moment
            'crossbar': (255, 0, 255),       # Magenta for crossbar
            'goal_line': (0, 165, 255),      # Orange for goal line
        }
        
        # Create directory for goal images
        os.makedirs(self.goal_images_dir, exist_ok=True)
        
        print("Hybrid ML + Hough goal detection system initialized")

    def detect_goalposts_ml(self, frame):
        """Detect goalposts using the Roboflow ML model"""
        try:
            result = self.roboflow_client.infer(frame, model_id=self.goalpost_model_id)
            current_detections = []
            
            if 'predictions' in result:
                for prediction in result['predictions']:
                    if prediction['class'] == 'goalpost':
                        x_center = prediction['x']
                        y_center = prediction['y']
                        width = prediction['width']
                        height = prediction['height']
                        confidence = prediction['confidence']
                        
                        # Convert center coordinates to bounding box
                        x1 = int(x_center - width / 2)
                        y1 = int(y_center - height / 2)
                        x2 = int(x_center + width / 2)
                        y2 = int(y_center + height / 2)
                        
                        # Ensure coordinates are within frame bounds
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        
                        detection = {
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'center': (int(x_center), int(y_center)),
                            'width': width,
                            'height': height
                        }
                        current_detections.append(detection)
            
            return current_detections
            
        except Exception as e:
            print(f"Error in goalpost detection: {e}")
            return []

    def refine_goal_structure_with_hough(self, frame, ml_detections):
        """
        Use Hough transform to refine goal structure within ML-detected regions
        """
        if len(ml_detections) < 2:
            return None
        
        # Create ROI around detected goalposts for Hough analysis
        all_x1 = [d['bbox'][0] for d in ml_detections]
        all_y1 = [d['bbox'][1] for d in ml_detections]
        all_x2 = [d['bbox'][2] for d in ml_detections]
        all_y2 = [d['bbox'][3] for d in ml_detections]
        
        # Expand ROI to include potential crossbar and goal area
        roi_x1 = max(0, min(all_x1) - 50)
        roi_y1 = max(0, min(all_y1) - 100)  # Look above posts for crossbar
        roi_x2 = min(frame.shape[1], max(all_x2) + 50)
        roi_y2 = min(frame.shape[0], max(all_y2) + 100)  # Look below for goal line
        
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if roi.size == 0:
            return None
        
        # Detect white lines in the ROI
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            mask_white,
            self.hough_params['rho'],
            self.hough_params['theta'],
            self.hough_params['threshold'],
            minLineLength=self.hough_params['min_line_length'],
            maxLineGap=self.hough_params['max_line_gap']
        )
        
        if lines is None:
            return None
        
        # Classify lines and find goal structure
        verticals, horizontals = self.classify_hough_lines(lines)
        
        # Convert lines back to full frame coordinates
        verticals_full = [(v[0] + roi_x1, v[1] + roi_y1, v[2] + roi_x1, v[3] + roi_y1) for v in verticals]
        horizontals_full = [(h[0] + roi_x1, h[1] + roi_y1, h[2] + roi_x1, h[3] + roi_y1) for h in horizontals]
        
        # Build goal structure from ML posts and Hough lines
        goal_structure = self.build_goal_from_ml_and_hough(ml_detections, verticals_full, horizontals_full, frame.shape)
        
        return goal_structure, verticals_full, horizontals_full, (roi_x1, roi_y1, roi_x2, roi_y2)

    def classify_hough_lines(self, lines):
        """Classify Hough lines as vertical or horizontal"""
        verticals = []
        horizontals = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            
            # Calculate angle
            if dx == 0:
                angle = 90
            else:
                angle = np.abs(np.degrees(np.arctan(dy / dx)))
            
            length = np.sqrt(dx**2 + dy**2)
            
            # Classify lines
            if 75 <= angle <= 105 and length > 30:  # Vertical lines
                verticals.append((x1, y1, x2, y2))
            elif (angle <= 15 or angle >= 165) and length > 50:  # Horizontal lines
                horizontals.append((x1, y1, x2, y2))
        
        return verticals, horizontals

    def build_goal_from_ml_and_hough(self, ml_detections, verticals, horizontals, frame_shape):
        """Build complete goal structure using ML posts and Hough lines"""
        if len(ml_detections) < 2:
            return None
        
        # Sort ML detections by x-position
        ml_detections.sort(key=lambda d: d['center'][0])
        left_post = ml_detections[0]
        right_post = ml_detections[-1]
        
        # Find the best crossbar from Hough lines
        best_crossbar = None
        best_crossbar_score = 0
        
        left_center = left_post['center']
        right_center = right_post['center']
        
        for h in horizontals:
            h_x1, h_y1, h_x2, h_y2 = h
            h_center_y = (h_y1 + h_y2) / 2
            
            # Check if horizontal line connects the posts
            left_dist = abs(h_x1 - left_center[0])
            right_dist = abs(h_x2 - right_center[0])
            height_diff_left = abs(h_center_y - left_center[1])
            height_diff_right = abs(h_center_y - right_center[1])
            
            # Score based on alignment with posts
            alignment_score = 200 - (height_diff_left + height_diff_right)
            connection_score = 100 - (left_dist + right_dist)
            length_score = min(abs(h_x2 - h_x1), 200)
            
            total_score = alignment_score + connection_score + length_score
            
            if total_score > best_crossbar_score:
                best_crossbar_score = total_score
                best_crossbar = h
        
        # Find goal line (horizontal line near bottom of posts)
        best_goal_line = None
        best_goal_line_score = 0
        
        left_bottom = left_post['bbox'][3]  # y2 of left post bbox
        right_bottom = right_post['bbox'][3]  # y2 of right post bbox
        
        for h in horizontals:
            h_x1, h_y1, h_x2, h_y2 = h
            h_center_y = (h_y1 + h_y2) / 2
            
            # Check if line is near bottom of posts and spans goal width
            left_dist = abs(h_x1 - left_center[0])
            right_dist = abs(h_x2 - right_center[0])
            height_diff_left = abs(h_center_y - left_bottom)
            height_diff_right = abs(h_center_y - right_bottom)
            
            if height_diff_left < 50 and height_diff_right < 50:
                goal_line_score = 200 - (left_dist + right_dist) + min(abs(h_x2 - h_x1), 200)
                
                if goal_line_score > best_goal_line_score:
                    best_goal_line_score = goal_line_score
                    best_goal_line = h
        
        # Build goal corners
        if best_crossbar:
            # Use crossbar for top, estimate bottom
            crossbar_x1, crossbar_y1, crossbar_x2, crossbar_y2 = best_crossbar
            top_left = (crossbar_x1, crossbar_y1)
            top_right = (crossbar_x2, crossbar_y2)
            
            if best_goal_line:
                # Use goal line for bottom
                goal_x1, goal_y1, goal_x2, goal_y2 = best_goal_line
                bottom_left = (goal_x1, goal_y1)
                bottom_right = (goal_x2, goal_y2)
            else:
                # Estimate bottom from posts
                bottom_y = max(left_bottom, right_bottom) + 20
                bottom_left = (left_center[0], bottom_y)
                bottom_right = (right_center[0], bottom_y)
        else:
            # Fallback: use ML posts to estimate goal area
            top_y = min(left_post['bbox'][1], right_post['bbox'][1]) - 50
            bottom_y = max(left_post['bbox'][3], right_post['bbox'][3]) + 20
            top_left = (left_center[0], top_y)
            top_right = (right_center[0], top_y)
            bottom_left = (left_center[0], bottom_y)
            bottom_right = (right_center[0], bottom_y)
        
        corners = (top_left, top_right, bottom_right, bottom_left)
        
        return {
            'corners': corners,
            'has_crossbar': best_crossbar is not None,
            'has_goal_line': best_goal_line is not None,
            'ml_posts': ml_detections,
            'score': best_crossbar_score + best_goal_line_score
        }

    def detect_goalpost(self, frame):
        """Hybrid goal detection using ML + Hough transform"""
        # Step 1: Detect goalposts with ML
        ml_detections = self.detect_goalposts_ml(frame)
        
        # Step 2: Refine goal structure with Hough transform
        goal_structure_result = self.refine_goal_structure_with_hough(frame, ml_detections)
        
        # Visualization
        debug_frame = frame.copy()
        
        # Draw ML detections
        for detection in ml_detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), self.colors['goalpost_ml'], 2)
            label = f"Goalpost: {confidence:.2f}"
            cv2.putText(debug_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['goalpost_ml'], 2)
            cv2.circle(debug_frame, detection['center'], 5, self.colors['goalpost_ml'], -1)

        if goal_structure_result:
            goal_structure, verticals, horizontals, roi = goal_structure_result
            roi_x1, roi_y1, roi_x2, roi_y2 = roi
            
            # Draw Hough lines
            for v in verticals:
                cv2.line(debug_frame, (v[0], v[1]), (v[2], v[3]), self.colors['goalpost_hough'], 2)
            for h in horizontals:
                color = self.colors['crossbar'] if goal_structure['has_crossbar'] and h == self.find_best_matching_line(horizontals, goal_structure['corners'][0][1]) else self.colors['goal_line']
                cv2.line(debug_frame, (h[0], h[1]), (h[2], h[3]), color, 2)
            
            # Draw ROI for debugging
            cv2.rectangle(debug_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (128, 128, 128), 1)
            
            # Draw goal area
            corners = goal_structure['corners']
            pts = np.array(corners, np.int32)
            
            overlay = debug_frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.drawContours(debug_frame, [pts], -1, self.colors['goal_area'], 3)
            debug_frame = cv2.addWeighted(overlay, 0.2, debug_frame, 0.8, 0)
            
            # Calculate homography
            H = self.compute_goal_homography(*corners)
            
            # Update state
            self.consecutive_goal_detections += 1
            if self.consecutive_goal_detections >= self.min_consecutive_detections:
                self.last_homography = H
                self.last_goal_corners = corners
                self.goal_detected = True
                
                goal_type = []
                if goal_structure['has_crossbar']:
                    goal_type.append("Crossbar")
                if goal_structure['has_goal_line']:
                    goal_type.append("Goal Line")
                
                cv2.putText(debug_frame, f"GOAL DETECTED ({', '.join(goal_type)})", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['goal_area'], 2)
                cv2.putText(debug_frame, f"ML Posts: {len(ml_detections)} Hough Lines: {len(verticals)+len(horizontals)}", 
                           (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['goal_area'], 2)
            else:
                self.goal_detected = False
                cv2.putText(debug_frame, f"Goal Stabilizing... {self.consecutive_goal_detections}/{self.min_consecutive_detections}", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            return debug_frame, H, corners
        else:
            self.consecutive_goal_detections = max(0, self.consecutive_goal_detections - 1)
            
            if self.consecutive_goal_detections > 0 and self.last_goal_corners:
                self.goal_detected = True
                cv2.putText(debug_frame, "GOAL (Using Previous)", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                return debug_frame, self.last_homography, self.last_goal_corners
            else:
                self.goal_detected = False
                cv2.putText(debug_frame, "NO GOAL DETECTED", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(debug_frame, f"ML Posts: {len(ml_detections)}", 
                           (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                return debug_frame, None, None

    def find_best_matching_line(self, lines, target_y):
        """Find the line closest to a target y-coordinate"""
        best_line = None
        best_diff = float('inf')
        
        for line in lines:
            y_center = (line[1] + line[3]) / 2
            diff = abs(y_center - target_y)
            if diff < best_diff:
                best_diff = diff
                best_line = line
        
        return best_line

    def compute_goal_homography(self, top_left, top_right, bottom_right, bottom_left):
        """Compute homography matrix for goal space"""
        src_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
        
        # Destination: 2m wide x 1m high goal space
        dst_pts = np.array([
            [0, 0],      # Top-left
            [2.0, 0],    # Top-right  
            [2.0, 1.0],  # Bottom-right
            [0.0, 1.0]   # Bottom-left
        ], dtype=np.float32)
        
        try:
            H, status = cv2.findHomography(src_pts, dst_pts)
            if H is not None and np.linalg.cond(H) < 1000:
                return H
        except Exception as e:
            print(f"Homography computation error: {e}")
        
        return None

    # [Keep all other methods the same: detect_ball_and_players, enhance_ball_detection, check_goal_scored, save_goal_moment_image, process_frame]

    def detect_ball_and_players(self, frame):
        """Detect ball and players using YOLO model"""
        results = self.model(frame, conf=0.25)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        ball_detections = detections[detections.class_id == self.BALL_CLASS_ID]
        player_detections = detections[detections.class_id == self.PERSON_CLASS_ID]
        
        ball_center = None
        ball_confidence = 0
        
        if len(ball_detections) > 0:
            best_ball_idx = np.argmax(ball_detections.confidence)
            x1, y1, x2, y2 = ball_detections.xyxy[best_ball_idx].astype(int)
            confidence = ball_detections.confidence[best_ball_idx]
            
            if confidence > 0.15:
                ball_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                ball_confidence = confidence
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['ball'], 2)
                cv2.circle(frame, ball_center, 8, self.colors['ball'], -1)
                label = f"Ball: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['ball'], 2)
        
        for i in range(len(player_detections)):
            x1, y1, x2, y2 = player_detections.xyxy[i].astype(int)
            confidence = player_detections.confidence[i]
            
            if confidence > 0.4:
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['player'], 2)
                label = f"Player: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['player'], 2)
        
        return frame, ball_center, ball_confidence

    def enhance_ball_detection(self, frame, ball_center):
        if ball_center is not None:
            return ball_center
            
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 80, 255])
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 80])
        
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        mask_ball = cv2.bitwise_or(mask_white, mask_black)
        
        kernel = np.ones((5,5), np.uint8)
        mask_ball = cv2.morphologyEx(mask_ball, cv2.MORPH_CLOSE, kernel)
        mask_ball = cv2.morphologyEx(mask_ball, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask_ball, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_contour = None
        best_circularity = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 2000:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:
                        if circularity > best_circularity:
                            best_circularity = circularity
                            best_contour = contour
        
        if best_contour is not None:
            M = cv2.moments(best_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                ball_center = (cx, cy)
                cv2.circle(frame, ball_center, 10, (0, 255, 255), 2)
                cv2.putText(frame, f"Enhanced Ball: {best_circularity:.2f}", 
                           (cx-60, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                return ball_center
        
        return None

    def check_goal_scored(self, ball_center, homography, frame_number):
        """Enhanced goal checking with better debugging"""
        if homography is None or ball_center is None:
            return False
        
        point = np.array([[ball_center]], dtype=np.float32)
        try:
            goal_space_pos = cv2.perspectiveTransform(point, homography)[0][0]
            x, y = goal_space_pos
            
            # More detailed debugging
            print(f"DEBUG: Ball at ({ball_center[0]}, {ball_center[1]}) -> Goal space: ({x:.3f}, {y:.3f})")
            print(f"DEBUG: Goal boundaries - X: 0.0 to 2.0, Y: 0.0 to 1.0")
            print(f"DEBUG: Ball is {'INSIDE' if (0 <= x <= 2.0) and (0 <= y <= 1.0) else 'OUTSIDE'} goal area")
            
            # Check if ball is within goal boundaries
            is_inside = (0 <= x <= 2.0) and (0 <= y <= 1.0)
            
            if is_inside and not self.goal_scored:
                self.goal_scored = True
                self.goal_frame_count = frame_number
                print(f"🎉 GOAL SCORED! Ball at goal position: ({x:.3f}m, {y:.3f}m) 🎉")
                return True
                
        except Exception as e:
            print(f"Error in perspective transform: {e}")
            
        return False

    def save_goal_moment_image(self, frame, frame_number, ball_center, homography):
        goal_frame = frame.copy()
        height, width = goal_frame.shape[:2]
        
        overlay = goal_frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), self.colors['goal_moment'], -1)
        goal_frame = cv2.addWeighted(overlay, 0.1, goal_frame, 0.9, 0)
        
        text = "GOAL MOMENT!"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
        text_x = (width - text_size[0]) // 2
        text_y = 100
        
        cv2.rectangle(goal_frame, 
                     (text_x - 20, text_y - text_size[1] - 20),
                     (text_x + text_size[0] + 20, text_y + 20),
                     (0, 0, 0), -1)
        cv2.putText(goal_frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 3, self.colors['goal_moment'], 5)
        
        info_text = f"Frame: {frame_number} | Ball Position: {ball_center}"
        cv2.putText(goal_frame, info_text, (50, height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(goal_frame, timestamp, (50, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if ball_center:
            cv2.circle(goal_frame, ball_center, 20, self.colors['goal_moment'], 3)
            cv2.circle(goal_frame, ball_center, 30, self.colors['goal_moment'], 2)
            cv2.circle(goal_frame, ball_center, 40, self.colors['goal_moment'], 1)
            cv2.putText(goal_frame, "SCORING BALL", 
                       (ball_center[0] - 60, ball_center[1] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['goal_moment'], 2)
        
        if self.last_goal_corners:
            pts = np.array(self.last_goal_corners, np.int32)
            cv2.polylines(goal_frame, [pts], True, self.colors['goal_moment'], 4)
            
            if homography is not None and ball_center is not None:
                point = np.array([[ball_center]], dtype=np.float32)
                try:
                    goal_space_pos = cv2.perspectiveTransform(point, homography)[0][0]
                    x, y = goal_space_pos
                    coord_text = f"Goal Position: ({x:.3f}m, {y:.3f}m)"
                    cv2.putText(goal_frame, coord_text, (width - 400, height - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                except:
                    pass
        
        filename = f"OUTPUT_{self.output_base_name}_goal_frame_{frame_number}.jpg"
        filepath = os.path.join(self.goal_images_dir, filename)
        cv2.imwrite(filepath, goal_frame)
        
        print(f"🎯 Goal moment saved: {filepath}")
        return filepath

    def process_frame(self, frame, frame_number, debug=False):
        frame_with_goal, homography, goal_corners = self.detect_goalpost(frame)
        frame_with_detections, ball_center, ball_confidence = self.detect_ball_and_players(frame_with_goal)
        
        if ball_center is None:
            ball_center = self.enhance_ball_detection(frame_with_detections, ball_center)
        
        goal_moment_image_path = None
        if homography is not None and ball_center is not None:
            is_new_goal = self.check_goal_scored(ball_center, homography, frame_number)
            
            if is_new_goal:
                goal_moment_image_path = self.save_goal_moment_image(
                    frame_with_detections, frame_number, ball_center, homography
                )
        
        if self.goal_scored:
            celebration_text = "GOAL!"
            text_size = cv2.getTextSize(celebration_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            
            cv2.putText(frame_with_detections, celebration_text, 
                       (text_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, self.colors['goal_text'], 5)
            cv2.putText(frame_with_detections, celebration_text, 
                       (text_x, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, self.colors['goal_text'], 5)
        
        # Enhanced status display
        status_y = frame.shape[0] - 30
        cv2.putText(frame_with_detections, f"Goal Detected: {self.goal_detected}", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   self.colors['goal_area'] if self.goal_detected else (0, 0, 255), 2)
        
        if ball_center:
            cv2.putText(frame_with_detections, f"Ball: ({ball_center[0]}, {ball_center[1]}) Conf: {ball_confidence:.2f}", 
                       (10, status_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['ball'], 2)
        
        # Enhanced goal space coordinates display
        if homography is not None and ball_center is not None:
            point = np.array([[ball_center]], dtype=np.float32)
            try:
                goal_space_pos = cv2.perspectiveTransform(point, homography)[0][0]
                x, y = goal_space_pos
                coord_text = f"Goal Space: ({x:.3f}, {y:.3f})"
                cv2.putText(frame_with_detections, coord_text, 
                           (10, status_y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Visual indicator with more precise boundaries
                if 0 <= x <= 2.0 and 0 <= y <= 1.0:
                    cv2.putText(frame_with_detections, "BALL IN GOAL! 🥅", 
                               (frame.shape[1] - 250, status_y - 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    # Show why ball is not in goal
                    if x < 0 or x > 2.0:
                        reason = "X out of bounds"
                    elif y < 0 or y > 1.0:
                        reason = "Y out of bounds"
                    else:
                        reason = "In goal area"
                    cv2.putText(frame_with_detections, f"Not goal: {reason}", 
                               (frame.shape[1] - 250, status_y - 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            except Exception as e:
                cv2.putText(frame_with_detections, f"Homography error", 
                           (10, status_y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        if self.goal_scored:
            status_text = f"STATUS: GOAL SCORED! (Frame {self.goal_frame_count})"
            cv2.putText(frame_with_detections, status_text, 
                       (frame.shape[1]//2 - 200, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['goal_text'], 3)
        
        return frame_with_detections, goal_moment_image_path

# [Keep argument parsing and main function the same]
def parse_arguments():
    parser = argparse.ArgumentParser(description='Hybrid ML + Hough Goal Detection System')
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('--roboflow-key', required=True, help='Roboflow API key for goalpost detection')
    parser.add_argument('--max-frames', type=int, default=500, 
                       help='Maximum number of frames to process (default: 500)')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Frame number to start processing from (default: 0)')
    parser.add_argument('--output-name', type=str, default=None,
                       help='Base name for output files (default: derived from input)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with detailed output')
    return parser.parse_args()

def get_output_base_name(input_video_path, custom_output_name=None):
    if custom_output_name:
        return custom_output_name
    base_name = os.path.basename(input_video_path)
    base_name = os.path.splitext(base_name)[0]
    base_name = os.path.basename(base_name)
    return base_name

def main():
    args = parse_arguments()
    output_base_name = get_output_base_name(args.input_video, args.output_name)
    system = SoccerAnalysisSystem(output_base_name, args.roboflow_key)
    
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.input_video}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video: {args.input_video}")
    print(f"Output base: {output_base_name}")
    print(f"Hybrid ML + Hough Goal Detection")
    print(f"Output video: {system.output_video_path}")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(system.output_video_path, fourcc, fps, (width, height))
    
    frame_count = args.start_frame
    processed_frames = 0
    goal_images_saved = []
    
    print("\nStarting hybrid goal detection (ML posts + Hough structure)...")
    
    try:
        while processed_frames < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if args.debug and processed_frames % 30 == 0:
                print(f"Processing frame {frame_count}...")
            
            processed_frame, goal_image_path = system.process_frame(frame, frame_count, args.debug)
            
            if goal_image_path:
                goal_images_saved.append(goal_image_path)
            
            out.write(processed_frame)
            frame_count += 1
            processed_frames += 1
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error during processing: {e}")
    
    cap.release()
    out.release()
    
    print(f"\nProcessing completed!")
    print(f"Processed {processed_frames} frames")
    print(f"Output video: {system.output_video_path}")
    
    if system.goal_scored:
        print(f"🎉 Goal was detected at frame {system.goal_frame_count}! 🎉")
        if goal_images_saved:
            print(f"Goal moment images saved:")
            for img_path in goal_images_saved:
                print(f"  - {img_path}")
    else:
        print("No goal was detected in the processed frames.")
        print("Check the output video to see if the goal area was properly defined.")

if __name__ == "__main__":
    main()