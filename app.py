import cv2
import numpy as np
import mediapipe as mp
import os
from math import sqrt

class RealisticDressFitting:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=True,  # Enable segmentation for better body masking
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_drawing = mp.solutions.drawing_utils

        # Dress parameters
        self.dresses = []
        self.current_dress_index = 0
        self.dress_scale = 1.0
        self.dress_position = [0, 0]
        self.auto_fit_enabled = True

        # Body measurements cache
        self.body_measurements = {}
        
        # Realism parameters
        self.shadow_intensity = 0.3
        self.lighting_direction = 1  # 1 for right, -1 for left
        self.body_contour_blur = 3

        # Load custom dresses
        self.load_custom_dresses()

    def load_custom_dresses(self):
        """Load dress images from assets/dresses/ folder with enhanced processing"""
        dress_dir = os.path.join(os.path.dirname(__file__), "assets", "dresses")

        if not os.path.exists(dress_dir):
            print(f"[WARNING] Dress folder not found: {dress_dir}")
            print("[INFO] Creating sample dresses...")
            self.create_sample_dresses()
            return

        dress_files = [f for f in os.listdir(dress_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        
        if not dress_files:
            print("[INFO] No dress images found, creating sample dresses...")
            self.create_sample_dresses()
            return

        for file in dress_files:
            path = os.path.join(dress_dir, file)
            dress = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if dress is None:
                print(f"[WARNING] Could not read {path}")
                continue
                
            # Ensure 4 channels (BGRA) for blending
            if dress.shape[2] == 3:
                dress = cv2.cvtColor(dress, cv2.COLOR_BGR2BGRA)
            
            # Enhance dress for realism
            dress = self.enhance_dress_texture(dress)
            self.dresses.append(dress)

        print(f"[INFO] Loaded {len(self.dresses)} dresses from {dress_dir}")

    def create_sample_dresses(self):
        """Create realistic sample dresses with textures"""
        for i in range(3):
            # Create larger, more detailed dress images
            dress = np.zeros((800, 400, 4), dtype=np.uint8)
            
            # Realistic dress colors with variations
            colors = [
                (120, 80, 200, 220),   # Purple silk
                (200, 100, 100, 210),  # Rose red
                (80, 150, 220, 215)    # Sky blue
            ]
            color = colors[i]
            
            # Create realistic dress shape with curved edges
            self.draw_realistic_dress(dress, color, i)
            self.add_dress_texture(dress, i)
            
            self.dresses.append(dress)

    def draw_realistic_dress(self, dress, color, style):
        """Draw realistic dress shape with proper contours"""
        h, w = dress.shape[:2]
        
        # Dress silhouette points
        if style == 0:  # A-line dress
            pts = np.array([
                [w//2 - 60, h//8],      # Left shoulder
                [w//2 + 60, h//8],      # Right shoulder
                [w//2 + 100, h - 50],   # Right hem
                [w//2 - 100, h - 50]    # Left hem
            ])
        elif style == 1:  # Fit and flare
            pts = np.array([
                [w//2 - 50, h//8],
                [w//2 + 50, h//8],
                [w//2 + 70, h//2],
                [w//2 + 120, h - 50],
                [w//2 - 120, h - 50],
                [w//2 - 70, h//2]
            ])
        else:  # Empire waist
            pts = np.array([
                [w//2 - 55, h//8],
                [w//2 + 55, h//8],
                [w//2 + 80, h//3],
                [w//2 + 60, h - 50],
                [w//2 - 60, h - 50],
                [w//2 - 80, h//3]
            ])
        
        # Fill dress body
        cv2.fillPoly(dress, [pts], color)
        
        # Add neckline
        neck_y = h//8
        cv2.ellipse(dress, (w//2, neck_y), (40, 20), 0, 0, 180, color, -1)
        
        # Add subtle folds
        self.add_dress_folds(dress, pts, color)

    def add_dress_texture(self, dress, style):
        """Add realistic fabric texture"""
        h, w = dress.shape[:2]
        
        # Create noise texture
        noise = np.random.randint(0, 15, (h, w), dtype=np.uint8)
        
        # Apply texture based on dress style
        if style == 0:  # Silk (smooth)
            texture_strength = 5
        elif style == 1:  # Cotton (medium texture)
            texture_strength = 8
        else:  # Linen (rough texture)
            texture_strength = 12
            
        # Blend texture with dress
        mask = dress[:, :, 3] > 0
        for c in range(3):
            dress[mask, c] = np.clip(
                dress[mask, c].astype(np.int16) + 
                noise[mask] - texture_strength//2, 0, 255
            ).astype(np.uint8)

    def add_dress_folds(self, dress, contour, color):
        """Add realistic fabric folds"""
        h, w = dress.shape[:2]
        
        # Add vertical folds
        for i in range(3):
            fold_x = w//4 + i * w//6
            fold_width = 8
            
            # Create gradient for fold shadow
            for x_offset in range(-fold_width, fold_width):
                alpha = 1.0 - abs(x_offset) / fold_width
                x_pos = fold_x + x_offset
                if 0 <= x_pos < w:
                    column_mask = (dress[:, x_pos, 3] > 0)
                    if np.any(column_mask):
                        adjustment = int(20 * (1 - alpha))  # Darken fold areas
                        for c in range(3):
                            dress[column_mask, x_pos, c] = np.clip(
                                dress[column_mask, x_pos, c].astype(np.int16) - adjustment, 0, 255
                            ).astype(np.uint8)

    def enhance_dress_texture(self, dress):
        """Enhance loaded dress images for realism"""
        # Add subtle noise for fabric texture
        noise = np.random.randint(-5, 5, dress.shape[:2], dtype=np.int16)
        mask = dress[:, :, 3] > 0
        
        for c in range(3):
            dress[mask, c] = np.clip(dress[mask, c].astype(np.int16) + noise[mask], 0, 255).astype(np.uint8)
        
        # Soften edges
        alpha = dress[:, :, 3]
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        dress[:, :, 3] = alpha
        
        return dress

    def calculate_body_measurements(self, landmarks, image_shape):
        """Calculate detailed body measurements for realistic fitting"""
        if landmarks is None:
            return {}
        
        h, w = image_shape[:2]
        
        def get_coord(landmark):
            return int(landmark.x * w), int(landmark.y * h)
        
        try:
            # Upper body points
            left_shoulder = get_coord(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
            right_shoulder = get_coord(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])
            left_hip = get_coord(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP])
            right_hip = get_coord(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP])
            
            # Lower body points
            left_knee = get_coord(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE])
            right_knee = get_coord(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE])
            
            # Calculate measurements
            shoulder_width = sqrt((right_shoulder[0] - left_shoulder[0])**2 + 
                                 (right_shoulder[1] - left_shoulder[1])**2)
            hip_width = sqrt((right_hip[0] - left_hip[0])**2 + 
                             (right_hip[1] - left_hip[1])**2)
            
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_y = (left_hip[1] + right_hip[1]) / 2
            
            # Calculate body proportions
            upper_body_height = hip_y - shoulder_y
            total_body_height = (max(left_knee[1], right_knee[1]) - shoulder_y) * 1.2  # Estimate full height
            
            return {
                'shoulder_width': shoulder_width,
                'hip_width': hip_width,
                'shoulder_y': shoulder_y,
                'hip_y': hip_y,
                'upper_body_height': upper_body_height,
                'total_body_height': total_body_height,
                'center_x': (left_shoulder[0] + right_shoulder[0]) // 2,
                'left_shoulder': left_shoulder,
                'right_shoulder': right_shoulder,
                'left_hip': left_hip,
                'right_hip': right_hip
            }
        except (IndexError, AttributeError):
            return {}

    def auto_fit_dress(self, body_measurements, dress_img):
        """Advanced auto-fitting considering body proportions"""
        if not body_measurements:
            return 1.0, [0, 0]

        dress_h, dress_w = dress_img.shape[:2]
        
        # Calculate scale based on multiple body measurements
        shoulder_scale = body_measurements['shoulder_width'] * 1.1 / dress_w
        height_scale = body_measurements['upper_body_height'] * 1.8 / dress_h
        
        # Weighted scale calculation
        scale = (shoulder_scale * 0.6 + height_scale * 0.4)
        scale = max(0.4, min(1.8, scale))

        # Precise positioning
        cx = body_measurements['center_x']
        shoulder_y = body_measurements['shoulder_y']
        
        x = cx - int((dress_w * scale) / 2)
        y = int(shoulder_y - 0.08 * dress_h * scale)  # Slight adjustment for neckline
        
        return scale, [x, y]

    def apply_realistic_lighting(self, dress_region, body_measurements, frame_region):
        """Apply realistic lighting and shadows to the dress"""
        if not body_measurements:
            return dress_region
            
        h, w = dress_region.shape[:2]
        
        # Create gradient shadow based on body shape
        shadow_map = np.ones((h, w), dtype=np.float32)
        
        # Left side shadow (assuming light from top-right)
        for x in range(w):
            shadow_intensity = 1.0 - (x / w) * self.shadow_intensity * self.lighting_direction
            shadow_map[:, x] = np.clip(shadow_intensity, 0.7, 1.0)
        
        # Apply shadow to dress
        for c in range(3):
            dress_region[:, :, c] = np.clip(
                dress_region[:, :, c].astype(np.float32) * shadow_map, 0, 255
            ).astype(np.uint8)
        
        return dress_region

    def blend_with_body_contours(self, dress_region, frame_region, alpha_channel):
        """Blend dress with body contours for natural look"""
        # Create soft mask
        soft_mask = cv2.GaussianBlur(alpha_channel, (self.body_contour_blur, self.body_contour_blur), 0)
        soft_mask = soft_mask / 255.0
        
        # Enhanced blending with body contours
        result = frame_region.copy().astype(np.float32)
        dress_float = dress_region.astype(np.float32)
        
        # Multiplicative blending for more natural look
        for c in range(3):
            result[:, :, c] = (
                soft_mask * dress_float[:, :, c] + 
                (1 - soft_mask) * result[:, :, c]
            )
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_dress(self, frame, dress_img, scale, position):
        """Enhanced dress application with realistic effects"""
        h, w = frame.shape[:2]
        dress_h, dress_w = dress_img.shape[:2]
        
        # Scale dress
        new_w = int(dress_w * scale)
        new_h = int(dress_h * scale)
        if new_w <= 0 or new_h <= 0:
            return frame.copy()

        dress_resized = cv2.resize(dress_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Calculate bounds
        x, y = position
        x1, y1, x2, y2 = x, y, x + new_w, y + new_h
        fx1, fy1 = max(0, x1), max(0, y1)
        fx2, fy2 = min(w, x2), min(h, y2)
        
        if fx1 >= fx2 or fy1 >= fy2:
            return frame.copy()

        # Calculate crop coordinates
        dx1, dy1 = fx1 - x1, fy1 - y1
        dx2, dy2 = dx1 + (fx2 - fx1), dy1 + (fy2 - fy1)

        # Extract regions
        roi = frame[fy1:fy2, fx1:fx2].copy()
        dress_crop = dress_resized[dy1:dy2, dx1:dx2].copy()
        
        if dress_crop.shape[2] == 4:
            alpha = dress_crop[:, :, 3]
            
            # Apply realistic lighting
            dress_crop = self.apply_realistic_lighting(dress_crop, self.body_measurements, roi)
            
            # Enhanced blending
            blended_roi = self.blend_with_body_contours(dress_crop, roi, alpha)
        else:
            # Fallback blending for images without alpha
            gray = cv2.cvtColor(dress_crop[:, :, :3], cv2.COLOR_BGR2GRAY)
            alpha = (gray > 10).astype(np.float32)
            blended_roi = self.blend_with_body_contours(dress_crop, roi, alpha * 255)

        # Apply final result
        result = frame.copy()
        result[fy1:fy2, fx1:fx2] = blended_roi
        return result

    def draw_measurement_info(self, frame, body_measurements):
        """Draw enhanced measurement visualization"""
        if not body_measurements:
            return
            
        # Draw shoulder points and line
        left_shoulder = body_measurements.get('left_shoulder')
        right_shoulder = body_measurements.get('right_shoulder')
        left_hip = body_measurements.get('left_hip')
        right_hip = body_measurements.get('right_hip')
        
        if left_shoulder and right_shoulder:
            # Draw body points
            cv2.circle(frame, left_shoulder, 6, (0, 255, 0), -1)
            cv2.circle(frame, right_shoulder, 6, (0, 255, 0), -1)
            cv2.line(frame, left_shoulder, right_shoulder, (0, 255, 0), 2)
            
            if left_hip and right_hip:
                cv2.circle(frame, left_hip, 6, (255, 0, 0), -1)
                cv2.circle(frame, right_hip, 6, (255, 0, 0), -1)
                cv2.line(frame, left_hip, right_hip, (255, 0, 0), 2)
            
            # Display measurements
            info_y = 150
            cv2.putText(frame, f"Shoulder: {body_measurements['shoulder_width']:.1f}px",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Hips: {body_measurements['hip_width']:.1f}px",
                       (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"Scale: {self.dress_scale:.2f}",
                       (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run(self):
        """Main application loop with enhanced UI"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("REALISTIC VIRTUAL DRESSING ROOM")
        print("CONTROLS:")
        print("N = Next dress")
        print("A = Toggle auto-fit")
        print("+/- = Scale adjustment") 
        print("Arrow Keys = Position adjustment")
        print("L = Change lighting direction")
        print("Q = Quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process pose
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                self.body_measurements = self.calculate_body_measurements(
                    results.pose_landmarks.landmark, frame.shape)
                
                # Smooth auto-fitting
                if self.auto_fit_enabled and self.body_measurements:
                    new_scale, new_pos = self.auto_fit_dress(
                        self.body_measurements, self.dresses[self.current_dress_index])
                    
                    SMOOTH = 0.15  # Reduced for more responsive fitting
                    self.dress_scale = (1 - SMOOTH) * self.dress_scale + SMOOTH * new_scale
                    self.dress_position[0] = int((1 - SMOOTH) * self.dress_position[0] + SMOOTH * new_pos[0])
                    self.dress_position[1] = int((1 - SMOOTH) * self.dress_position[1] + SMOOTH * new_pos[1])

            # Apply dress if available
            if self.dresses:
                frame = self.apply_dress(
                    frame,
                    self.dresses[self.current_dress_index],
                    self.dress_scale,
                    self.dress_position
                )

            # Draw measurement info
            if self.body_measurements:
                self.draw_measurement_info(frame, self.body_measurements)

            # Enhanced UI display
            cv2.putText(frame, f"Dress {self.current_dress_index + 1}/{len(self.dresses)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Auto-fit: {'ON' if self.auto_fit_enabled else 'OFF'}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Lighting: {'Right' if self.lighting_direction > 0 else 'Left'}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Realistic Virtual Dressing Room", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n') and self.dresses:
                self.current_dress_index = (self.current_dress_index + 1) % len(self.dresses)
                print(f"Switched to dress {self.current_dress_index + 1}")
            elif key == ord('a'):
                self.auto_fit_enabled = not self.auto_fit_enabled
                print(f"Auto-fit: {'ON' if self.auto_fit_enabled else 'OFF'}")
            elif key == ord('l'):
                self.lighting_direction *= -1
                print(f"Lighting direction changed")
            elif key in (ord('+'), ord('=')):
                self.dress_scale = min(2.0, self.dress_scale + 0.03)
                self.auto_fit_enabled = False
            elif key == ord('-'):
                self.dress_scale = max(0.3, self.dress_scale - 0.03)
                self.auto_fit_enabled = False
            elif key == 81:  # Left
                self.dress_position[0] -= 3
                self.auto_fit_enabled = False
            elif key == 83:  # Right
                self.dress_position[0] += 3
                self.auto_fit_enabled = False
            elif key == 82:  # Up
                self.dress_position[1] -= 3
                self.auto_fit_enabled = False
            elif key == 84:  # Down
                self.dress_position[1] += 3
                self.auto_fit_enabled = False

        cap.release()
        cv2.destroyAllWindows()

def main():
    dressing_room = RealisticDressFitting()
    dressing_room.run()

if __name__ == "__main__":
    main()