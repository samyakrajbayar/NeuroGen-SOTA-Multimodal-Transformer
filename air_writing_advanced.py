#!/usr/bin/env python3
"""
Advanced Air Writing System
===========================

An AI-powered gesture recognition system for writing in the air using hand tracking.
Features multiple writing modes, gesture controls, shape recognition, and OCR.

Author: AI Assistant
Version: 2.0.0
License: MIT
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
import os
import json
from datetime import datetime
from enum import Enum
import argparse
import colorsys

# Optional imports for advanced features
try:
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Some smoothing features disabled.")

try:
    import tkinter as tk
    from tkinter import colorchooser, filedialog, messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("Warning: tkinter not available. GUI features disabled.")


class WritingMode(Enum):
    """Enumeration of available writing modes."""
    PEN = "pen"
    BRUSH = "brush"
    NEON = "neon"
    LASER = "laser"
    SPRAY = "spray"
    ERASER = "eraser"


class GestureType(Enum):
    """Enumeration of recognized gesture types."""
    NONE = "none"
    WRITE = "write"
    ERASE = "erase"
    SELECT = "select"
    CLEAR = "clear"
    SAVE = "save"
    UNDO = "undo"
    HOVER = "hover"
    MENU = "menu"
    PALM = "palm"


class ColorPalette:
    """Predefined color palette with easy access."""
    COLORS = {
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'cyan': (255, 255, 0),
        'magenta': (255, 0, 255),
        'orange': (0, 165, 255),
        'purple': (128, 0, 128),
        'pink': (147, 20, 255),
        'lime': (50, 205, 50),
        'gold': (0, 215, 255),
        'silver': (192, 192, 192),
        'brown': (42, 42, 165),
        'navy': (128, 0, 0),
    }
    
    @classmethod
    def get_color_list(cls):
        """Return list of available colors."""
        return list(cls.COLORS.values())
    
    @classmethod
    def get_rainbow_color(cls, position, total=100):
        """Get rainbow color at position."""
        hue = position / total
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return tuple(int(x * 255) for x in rgb[::-1])  # Convert to BGR


class KalmanFilter:
    """Simple Kalman filter for smoothing hand trajectory."""
    
    def __init__(self, process_noise=1e-5, measurement_noise=1e-2):
        self.x = np.array([0., 0.])  # State
        self.P = np.eye(2)  # Error covariance
        self.Q = np.eye(2) * process_noise  # Process noise
        self.R = np.eye(2) * measurement_noise  # Measurement noise
        self.H = np.eye(2)  # Measurement matrix
        self.F = np.eye(2)  # State transition matrix
        
    def predict(self):
        """Predict next state."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x
    
    def update(self, measurement):
        """Update state with measurement."""
        z = np.array(measurement)
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        return self.x
    
    def filter(self, measurement):
        """Full filter step."""
        self.predict()
        return self.update(measurement)


class GestureRecognizer:
    """Advanced gesture recognition using hand landmarks."""
    
    def __init__(self, swipe_threshold=30, hold_duration=1.0):
        self.swipe_threshold = swipe_threshold
        self.hold_duration = hold_duration
        self.position_history = deque(maxlen=30)
        self.gesture_start_time = None
        self.last_gesture = GestureType.NONE
        self.swipe_start_pos = None
        
    def detect_gesture(self, hand_landmarks, prev_gesture=None):
        """
        Detect gesture from hand landmarks.
        Returns: (gesture_type, confidence, metadata)
        """
        if hand_landmarks is None:
            return GestureType.NONE, 0.0, {}
        
        # Get finger states (extended or not)
        fingers = self._get_finger_states(hand_landmarks)
        
        # Get index finger tip position
        index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        current_pos = (int(index_tip.x * 1280), int(index_tip.y * 720))
        self.position_history.append(current_pos)
        
        # Detect gesture based on finger configuration
        gesture = self._classify_finger_gesture(fingers)
        confidence = 1.0
        metadata = {'position': current_pos, 'fingers': fingers}
        
        # Detect swipes and holds
        if len(self.position_history) >= 10:
            swipe = self._detect_swipe()
            if swipe:
                metadata['swipe'] = swipe
                if gesture == GestureType.WRITE:
                    if swipe == 'left':
                        gesture = GestureType.UNDO
                    elif swipe == 'right':
                        gesture = GestureType.SAVE
        
        # Check for hold gestures
        if gesture != self.last_gesture:
            self.gesture_start_time = time.time()
            self.swipe_start_pos = current_pos
        else:
            hold_time = time.time() - self.gesture_start_time if self.gesture_start_time else 0
            metadata['hold_time'] = hold_time
            
            # Clear canvas on palm hold
            if gesture == GestureType.PALM and hold_time > self.hold_duration:
                gesture = GestureType.CLEAR
        
        self.last_gesture = gesture
        return gesture, confidence, metadata
    
    def _get_finger_states(self, hand_landmarks):
        """Determine which fingers are extended."""
        fingers = []
        
        # Thumb (check x distance from pinky base)
        thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_IP]
        fingers.append(1 if thumb_tip.x < thumb_ip.x else 0)
        
        # Other 4 fingers (check y position)
        finger_tips = [
            mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
            mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
            mp.solutions.hands.HandLandmark.PINKY_TIP
        ]
        finger_pips = [
            mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP,
            mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP,
            mp.solutions.hands.HandLandmark.RING_FINGER_PIP,
            mp.solutions.hands.HandLandmark.PINKY_PIP
        ]
        
        for tip, pip in zip(finger_tips, finger_pips):
            fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y else 0)
        
        return fingers
    
    def _classify_finger_gesture(self, fingers):
        """Classify gesture based on finger states."""
        # fingers = [thumb, index, middle, ring, pinky]
        
        if fingers == [0, 1, 0, 0, 0]:
            return GestureType.WRITE  # Index only
        elif fingers == [0, 1, 1, 0, 0]:
            return GestureType.ERASE  # Index + Middle
        elif fingers == [1, 1, 1, 1, 1]:
            return GestureType.PALM  # Open hand
        elif fingers == [0, 1, 1, 1, 1]:
            return GestureType.MENU  # 4 fingers
        elif fingers == [0, 0, 0, 0, 0]:
            return GestureType.HOVER  # Fist
        elif fingers == [1, 0, 0, 0, 0]:
            return GestureType.SAVE  # Thumb only
        elif fingers == [0, 0, 0, 0, 1]:
            return GestureType.UNDO  # Pinky only
        elif fingers == [1, 1, 1, 0, 0]:
            return GestureType.SELECT  # Thumb + Index + Middle
        else:
            return GestureType.HOVER
    
    def _detect_swipe(self):
        """Detect swipe gestures from position history."""
        if len(self.position_history) < 10:
            return None
        
        start_pos = self.position_history[0]
        end_pos = self.position_history[-1]
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        if abs(dx) > abs(dy) and abs(dx) > self.swipe_threshold:
            return 'right' if dx > 0 else 'left'
        elif abs(dy) > self.swipe_threshold:
            return 'down' if dy > 0 else 'up'
        
        return None


class BrushEngine:
    """Advanced brush engine with multiple drawing modes."""
    
    def __init__(self, canvas_size=(1280, 720)):
        self.canvas_size = canvas_size
        self.canvas = np.zeros((*canvas_size, 3), dtype=np.uint8)
        self.temp_canvas = np.zeros((*canvas_size, 3), dtype=np.uint8)
        self.mode = WritingMode.PEN
        self.color = (0, 255, 0)  # Green default
        self.size = 5
        self.opacity = 255
        self.smoothing = 5
        self.rainbow_mode = False
        self.rainbow_position = 0
        
        # Brush history for undo/redo
        self.history = []
        self.history_index = -1
        self.max_history = 20
        
        # Kalman filter for smoothing
        self.kalman = KalmanFilter()
        self.last_point = None
        
        # Brush parameters
        self.brush_params = {
            WritingMode.PEN: {'min_size': 1, 'max_size': 20, 'smooth': True},
            WritingMode.BRUSH: {'min_size': 3, 'max_size': 50, 'pressure': True},
            WritingMode.NEON: {'min_size': 5, 'max_size': 30, 'glow': True},
            WritingMode.LASER: {'min_size': 2, 'max_size': 15, 'trail': True},
            WritingMode.SPRAY: {'min_size': 10, 'max_size': 100, 'density': 50},
            WritingMode.ERASER: {'min_size': 10, 'max_size': 100},
        }
        
        # Spray particles
        self.spray_particles = []
        
    def set_mode(self, mode):
        """Set writing mode."""
        if isinstance(mode, str):
            mode = WritingMode(mode)
        self.mode = mode
        self.last_point = None
        
    def set_color(self, color):
        """Set brush color."""
        self.color = color
        self.rainbow_mode = False
        
    def set_size(self, size):
        """Set brush size."""
        params = self.brush_params.get(self.mode, {})
        min_size = params.get('min_size', 1)
        max_size = params.get('max_size', 50)
        self.size = max(min_size, min(max_size, size))
        
    def save_state(self):
        """Save current canvas state for undo."""
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        
        self.history.append(self.canvas.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)
        else:
            self.history_index += 1
    
    def undo(self):
        """Undo last action."""
        if self.history_index > 0:
            self.history_index -= 1
            self.canvas = self.history[self.history_index].copy()
            return True
        return False
    
    def redo(self):
        """Redo last undone action."""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.canvas = self.history[self.history_index].copy()
            return True
        return False
    
    def clear(self):
        """Clear canvas."""
        self.save_state()
        self.canvas = np.zeros((*self.canvas_size, 3), dtype=np.uint8)
        
    def draw(self, point, prev_point=None):
        """Draw at specified point."""
        if point is None:
            self.last_point = None
            return
        
        # Apply smoothing
        if self.smoothing > 0 and SCIPY_AVAILABLE and prev_point is not None:
            smoothed = self._smooth_point(point, prev_point)
        else:
            smoothed = point
        
        # Get rainbow color if enabled
        if self.rainbow_mode:
            self.color = ColorPalette.get_rainbow_color(self.rainbow_position)
            self.rainbow_position = (self.rainbow_position + 1) % 100
        
        # Draw based on mode
        if self.mode == WritingMode.PEN:
            self._draw_pen(smoothed, prev_point)
        elif self.mode == WritingMode.BRUSH:
            self._draw_brush(smoothed, prev_point)
        elif self.mode == WritingMode.NEON:
            self._draw_neon(smoothed, prev_point)
        elif self.mode == WritingMode.LASER:
            self._draw_laser(smoothed, prev_point)
        elif self.mode == WritingMode.SPRAY:
            self._draw_spray(smoothed)
        elif self.mode == WritingMode.ERASER:
            self._draw_eraser(smoothed, prev_point)
        
        self.last_point = smoothed
    
    def _smooth_point(self, point, prev_point):
        """Apply smoothing to point."""
        if prev_point is None:
            return point
        
        # Simple moving average
        alpha = 1.0 / self.smoothing
        x = int(alpha * point[0] + (1 - alpha) * prev_point[0])
        y = int(alpha * point[1] + (1 - alpha) * prev_point[1])
        return (x, y)
    
    def _draw_pen(self, point, prev_point):
        """Draw with pen mode."""
        if prev_point is not None:
            cv2.line(self.canvas, prev_point, point, self.color, self.size, cv2.LINE_AA)
        else:
            cv2.circle(self.canvas, point, self.size // 2, self.color, -1, cv2.LINE_AA)
    
    def _draw_brush(self, point, prev_point):
        """Draw with brush mode (variable width)."""
        if prev_point is not None:
            # Calculate speed for pressure simulation
            distance = np.sqrt((point[0] - prev_point[0])**2 + (point[1] - prev_point[1])**2)
            speed = min(distance / 10.0, 1.0)
            
            # Adjust size based on speed (slower = thicker)
            dynamic_size = int(self.size * (1.5 - speed * 0.5))
            
            # Draw tapered stroke
            cv2.line(self.canvas, prev_point, point, self.color, dynamic_size, cv2.LINE_AA)
        else:
            cv2.circle(self.canvas, point, self.size, self.color, -1, cv2.LINE_AA)
    
    def _draw_neon(self, point, prev_point):
        """Draw with neon glow effect."""
        # Create glow layers
        glow_size = self.size * 3
        
        # Draw main stroke
        if prev_point is not None:
            cv2.line(self.canvas, prev_point, point, self.color, self.size, cv2.LINE_AA)
            
            # Draw glow
            for i, alpha in enumerate([0.3, 0.2, 0.1]):
                glow_layer = np.zeros_like(self.canvas)
                glow_thickness = self.size + (i + 1) * 3
                cv2.line(glow_layer, prev_point, point, self.color, glow_thickness, cv2.LINE_AA)
                self.canvas = cv2.addWeighted(self.canvas, 1.0, glow_layer, alpha, 0)
        else:
            cv2.circle(self.canvas, point, self.size, self.color, -1, cv2.LINE_AA)
    
    def _draw_laser(self, point, prev_point):
        """Draw with laser mode."""
        # Draw bright core
        if prev_point is not None:
            cv2.line(self.canvas, prev_point, point, (255, 255, 255), max(1, self.size // 3), cv2.LINE_AA)
            cv2.line(self.canvas, prev_point, point, self.color, self.size, cv2.LINE_AA)
        else:
            cv2.circle(self.canvas, point, self.size // 2, self.color, -1, cv2.LINE_AA)
    
    def _draw_spray(self, point, prev_point=None):
        """Draw with spray paint effect."""
        # Generate random spray particles
        n_particles = self.size
        for _ in range(n_particles):
            # Random offset from center
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, self.size)
            offset_x = int(radius * np.cos(angle))
            offset_y = int(radius * np.sin(angle))
            
            spray_point = (point[0] + offset_x, point[1] + offset_y)
            
            # Check bounds
            if 0 <= spray_point[0] < self.canvas_size[0] and 0 <= spray_point[1] < self.canvas_size[1]:
                # Random opacity
                alpha = np.random.uniform(0.3, 0.8)
                color = tuple(int(c * alpha) for c in self.color)
                self.canvas[spray_point[1], spray_point[0]] = color
    
    def _draw_eraser(self, point, prev_point):
        """Draw with eraser."""
        if prev_point is not None:
            cv2.line(self.canvas, prev_point, point, (0, 0, 0), self.size * 2, cv2.LINE_AA)
        else:
            cv2.circle(self.canvas, point, self.size, (0, 0, 0), -1, cv2.LINE_AA)
    
    def get_canvas(self):
        """Get current canvas."""
        return self.canvas.copy()
    
    def save(self, filepath):
        """Save canvas to file."""
        cv2.imwrite(filepath, self.canvas)
        
    def load(self, filepath):
        """Load image as background."""
        img = cv2.imread(filepath)
        if img is not None:
            self.save_state()
            self.canvas = cv2.resize(img, self.canvas_size)


class AirWritingSystem:
    """Main Air Writing System class."""
    
    def __init__(self, args=None):
        # Parse arguments
        self.args = args or self._parse_arguments()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.args.camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual camera resolution
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize components
        self.gesture_recognizer = GestureRecognizer()
        self.brush_engine = BrushEngine(canvas_size=(self.width, self.height))
        
        # State variables
        self.running = True
        self.show_landmarks = False
        self.show_help = False
        self.menu_active = False
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()
        self.current_gesture = GestureType.NONE
        self.gesture_confidence = 0.0
        self.gesture_metadata = {}
        
        # Color palette
        self.colors = ColorPalette.get_color_list()
        self.color_index = 1  # Start with green
        self.brush_engine.set_color(self.colors[self.color_index])
        
        # Initialize
        self.brush_engine.save_state()
        
    def _parse_arguments(self):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description='Advanced Air Writing System')
        parser.add_argument('--mode', type=str, default='pen',
                          choices=['pen', 'brush', 'neon', 'laser', 'spray'],
                          help='Writing mode')
        parser.add_argument('--camera', type=int, default=0,
                          help='Camera device ID')
        parser.add_argument('--fps', type=int, default=30,
                          help='Target FPS')
        parser.add_argument('--smooth', type=int, default=5,
                          help='Smoothing level (0-10)')
        parser.add_argument('--mirror', action='store_true',
                          help='Enable mirror mode')
        return parser.parse_args()
    
    def process_frame(self, frame):
        """Process single frame."""
        # Flip frame if mirror mode
        if self.args.mirror:
            frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands
        results = self.hands.process(rgb_frame)
        
        # Detect gesture and draw
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Detect gesture
                gesture, confidence, metadata = self.gesture_recognizer.detect_gesture(
                    hand_landmarks, self.current_gesture
                )
                self.current_gesture = gesture
                self.gesture_confidence = confidence
                self.gesture_metadata = metadata
                
                # Get index finger tip position
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_tip.x * self.width)
                y = int(index_tip.y * self.height)
                
                # Handle gestures
                self._handle_gesture(gesture, (x, y))
                
                # Draw landmarks if enabled
                if self.show_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
        else:
            self.current_gesture = GestureType.NONE
            self.brush_engine.last_point = None
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.fps_history.append(fps)
        self.last_time = current_time
        
        return frame
    
    def _handle_gesture(self, gesture, position):
        """Handle detected gesture."""
        if gesture == GestureType.WRITE:
            self.brush_engine.draw(position, self.brush_engine.last_point)
            
        elif gesture == GestureType.ERASE:
            prev_mode = self.brush_engine.mode
            self.brush_engine.set_mode(WritingMode.ERASER)
            self.brush_engine.draw(position, self.brush_engine.last_point)
            self.brush_engine.set_mode(prev_mode)
            
        elif gesture == GestureType.CLEAR:
            if 'hold_time' in self.gesture_metadata and self.gesture_metadata['hold_time'] > 1.0:
                self.brush_engine.clear()
                print("Canvas cleared!")
                
        elif gesture == GestureType.SAVE:
            self.save_canvas()
            
        elif gesture == GestureType.UNDO:
            self.brush_engine.undo()
    
    def render(self, frame):
        """Render final output."""
        # Get canvas and overlay on frame
        canvas = self.brush_engine.get_canvas()
        
        # Blend canvas with frame
        alpha = 1.0
        output = cv2.addWeighted(frame, 1 - alpha, canvas, alpha, 0)
        
        # Add UI overlay
        self._draw_ui(output)
        
        return output
    
    def _draw_ui(self, frame):
        """Draw user interface overlay."""
        # Draw FPS
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        cv2.putText(frame, f"FPS: {int(avg_fps)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw current mode
        mode_text = f"Mode: {self.brush_engine.mode.value.upper()}"
        cv2.putText(frame, mode_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.brush_engine.color, 2)
        
        # Draw current gesture
        gesture_text = f"Gesture: {self.current_gesture.value}"
        cv2.putText(frame, gesture_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw color indicator
        color_rect_size = 30
        cv2.rectangle(frame, (10, 110), (10 + color_rect_size, 110 + color_rect_size),
                     self.brush_engine.color, -1)
        cv2.rectangle(frame, (10, 110), (10 + color_rect_size, 110 + color_rect_size),
                     (255, 255, 255), 2)
        
        # Draw brush size
        cv2.putText(frame, f"Size: {self.brush_engine.size}", (50, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw instructions if help is enabled
        if self.show_help:
            self._draw_help(frame)
        
        # Draw color palette
        self._draw_color_palette(frame)
    
    def _draw_help(self, frame):
        """Draw help text."""
        help_text = [
            "Controls:",
            "Q/ESC - Quit",
            "S - Save canvas",
            "C - Clear canvas",
            "Z - Undo",
            "B - Change mode",
            "+/- - Brush size",
            "H - Toggle help",
            "",
            "Gestures:",
            "Index finger - Write",
            "Two fingers - Erase",
            "Open palm 2s - Clear",
            "Thumb up - Save"
        ]
        
        y = 200
        for line in help_text:
            cv2.putText(frame, line, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 25
    
    def _draw_color_palette(self, frame):
        """Draw color palette at bottom."""
        palette_y = self.height - 50
        box_size = 30
        spacing = 10
        
        for i, color in enumerate(self.colors[:10]):
            x = 10 + i * (box_size + spacing)
            cv2.rectangle(frame, (x, palette_y), (x + box_size, palette_y + box_size),
                         color, -1)
            
            # Highlight selected color
            if i == self.color_index:
                cv2.rectangle(frame, (x - 2, palette_y - 2),
                            (x + box_size + 2, palette_y + box_size + 2),
                            (255, 255, 255), 2)
    
    def handle_key(self, key):
        """Handle keyboard input."""
        if key == ord('q') or key == 27:  # Q or ESC
            self.running = False
            
        elif key == ord('s'):  # Save
            self.save_canvas()
            
        elif key == ord('c'):  # Clear
            self.brush_engine.clear()
            
        elif key == ord('z'):  # Undo
            self.brush_engine.undo()
            
        elif key == ord('y'):  # Redo
            self.brush_engine.redo()
            
        elif key == ord('b'):  # Change mode
            modes = list(WritingMode)
            current_idx = modes.index(self.brush_engine.mode)
            next_idx = (current_idx + 1) % len(modes)
            self.brush_engine.set_mode(modes[next_idx])
            
        elif key == ord('e'):  # Toggle eraser
            if self.brush_engine.mode == WritingMode.ERASER:
                self.brush_engine.set_mode(WritingMode.PEN)
            else:
                self.brush_engine.set_mode(WritingMode.ERASER)
                
        elif key == ord('+'):  # Increase size
            self.brush_engine.set_size(self.brush_engine.size + 2)
            
        elif key == ord('-'):  # Decrease size
            self.brush_engine.set_size(self.brush_engine.size - 2)
            
        elif key == ord('h'):  # Toggle help
            self.show_help = not self.show_help
            
        elif key == ord('l'):  # Toggle landmarks
            self.show_landmarks = not self.show_landmarks
            
        elif key == ord('r'):  # Rainbow mode
            self.brush_engine.rainbow_mode = not self.brush_engine.rainbow_mode
            
        elif ord('0') <= key <= ord('9'):  # Color selection
            idx = key - ord('0')
            if idx < len(self.colors):
                self.color_index = idx
                self.brush_engine.set_color(self.colors[idx])
    
    def save_canvas(self):
        """Save canvas to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"air_writing_{timestamp}.png"
        self.brush_engine.save(filename)
        print(f"Canvas saved to: {filename}")
    
    def run(self):
        """Main run loop."""
        print("=" * 50)
        print("Advanced Air Writing System v2.0")
        print("=" * 50)
        print("\nControls:")
        print("  Q/ESC - Quit")
        print("  S - Save canvas")
        print("  C - Clear canvas")
        print("  Z - Undo")
        print("  B - Change writing mode")
        print("  +/- - Adjust brush size")
        print("  H - Toggle help")
        print("  L - Toggle landmarks")
        print("  R - Toggle rainbow mode")
        print("  0-9 - Select color")
        print("\nGestures:")
        print("  Index finger - Write/Draw")
        print("  Two fingers - Erase")
        print("  Open palm (hold 2s) - Clear canvas")
        print("  Thumb up - Save canvas")
        print("\nPress 'H' in app for full help")
        print("=" * 50)
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Render output
            output = self.render(processed_frame)
            
            # Show output
            cv2.imshow("Air Writing System", output)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            self.handle_key(key)
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("\nThank you for using Air Writing System!")


def main():
    """Main entry point."""
    try:
        app = AirWritingSystem()
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
