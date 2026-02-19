#!/usr/bin/env python3
"""
AI Character Recognition Module
===============================

Handwriting recognition using deep learning.
Supports real-time OCR for air writing system.

Author: AI Assistant
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os
import pickle
from collections import deque
import time


class CharacterRecognizer:
    """Deep learning based character recognition."""
    
    def __init__(self, model_path=None, confidence_threshold=0.85):
        self.confidence_threshold = confidence_threshold
        self.input_shape = (28, 28, 1)
        self.classes = self._get_character_classes()
        self.model = None
        self.stroke_history = deque(maxlen=100)
        self.is_recognizing = False
        self.recognition_buffer = []
        
        # Try to load pre-trained model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("No pre-trained model found. Building new model...")
            self._build_model()
    
    def _get_character_classes(self):
        """Define character classes (A-Z, a-z, 0-9)."""
        classes = []
        # Uppercase A-Z
        classes.extend([chr(i) for i in range(65, 91)])
        # Lowercase a-z
        classes.extend([chr(i) for i in range(97, 123)])
        # Digits 0-9
        classes.extend([str(i) for i in range(10)])
        # Common symbols
        classes.extend(['+', '-', '×', '÷', '=', '?', '!', '.', ','])
        return classes
    
    def _build_model(self):
        """Build CNN model for character recognition."""
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Fully connected layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(len(self.classes), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print(f"Model built with {len(self.classes)} output classes")
    
    def preprocess_image(self, image):
        """
        Preprocess image for recognition.
        
        Args:
            image: Input image (BGR from OpenCV)
        
        Returns:
            Preprocessed image ready for model
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to 28x28
        resized = cv2.resize(gray, (28, 28))
        
        # Normalize
        normalized = resized.astype('float32') / 255.0
        
        # Reshape for model input
        reshaped = normalized.reshape(1, 28, 28, 1)
        
        return reshaped
    
    def predict(self, image):
        """
        Predict character from image.
        
        Args:
            image: Input image containing character
        
        Returns:
            tuple: (predicted_char, confidence, all_predictions)
        """
        if self.model is None:
            return None, 0.0, {}
        
        # Preprocess
        processed = self.preprocess_image(image)
        
        # Predict
        predictions = self.model.predict(processed, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        
        # Get top 5 predictions
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
        top_5 = {self.classes[i]: float(predictions[0][i]) 
                for i in top_5_indices}
        
        if confidence >= self.confidence_threshold:
            return self.classes[predicted_idx], confidence, top_5
        else:
            return None, confidence, top_5
    
    def extract_character_region(self, canvas, bbox_margin=20):
        """
        Extract character region from canvas.
        
        Args:
            canvas: Drawing canvas
            bbox_margin: Margin around bounding box
        
        Returns:
            Extracted character image or None
        """
        # Convert to grayscale
        if len(canvas.shape) == 3:
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        else:
            gray = canvas
        
        # Threshold to get binary image
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add margin
        x = max(0, x - bbox_margin)
        y = max(0, y - bbox_margin)
        w = w + 2 * bbox_margin
        h = h + 2 * bbox_margin
        
        # Extract region
        char_region = gray[y:y+h, x:x+w]
        
        return char_region
    
    def recognize_canvas(self, canvas):
        """
        Recognize character from canvas.
        
        Args:
            canvas: Drawing canvas with character
        
        Returns:
            Recognition result dictionary
        """
        # Extract character region
        char_region = self.extract_character_region(canvas)
        
        if char_region is None:
            return {
                'character': None,
                'confidence': 0.0,
                'success': False,
                'alternatives': {}
            }
        
        # Predict
        char, confidence, alternatives = self.predict(char_region)
        
        return {
            'character': char,
            'confidence': confidence,
            'success': char is not None,
            'alternatives': alternatives,
            'image': char_region
        }
    
    def train(self, x_train, y_train, x_val, y_val, epochs=50, batch_size=128):
        """
        Train the model.
        
        Args:
            x_train: Training images
            y_train: Training labels
            x_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if self.model is None:
            self._build_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def save_model(self, filepath):
        """Save model to file."""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file."""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def evaluate(self, x_test, y_test):
        """Evaluate model on test set."""
        if self.model is None:
            print("No model loaded")
            return None
        
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        return {'loss': loss, 'accuracy': accuracy}


class ShapeRecognizer:
    """Recognize and perfect geometric shapes."""
    
    def __init__(self):
        self.shapes = ['circle', 'rectangle', 'triangle', 'line', 'ellipse']
        self.min_points = 20
        self.similarity_threshold = 0.8
    
    def recognize(self, points):
        """
        Recognize shape from points.
        
        Args:
            points: List of (x, y) points
        
        Returns:
            dict with shape type and parameters
        """
        if len(points) < self.min_points:
            return None
        
        points = np.array(points)
        
        # Calculate features
        area = cv2.contourArea(points.astype(np.float32))
        perimeter = cv2.arcLength(points.astype(np.float32), True)
        
        if perimeter == 0:
            return None
        
        # Circularness: 4π*area/perimeter² (1 for perfect circle)
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # Check for circle
        if circularity > 0.7:
            center, radius = cv2.minEnclosingCircle(points.astype(np.float32))
            return {
                'shape': 'circle',
                'center': (int(center[0]), int(center[1])),
                'radius': int(radius),
                'confidence': circularity
            }
        
        # Check for rectangle using bounding box
        rect = cv2.minAreaRect(points.astype(np.float32))
        box = cv2.boxPoints(rect)
        box_area = cv2.contourArea(box)
        
        if box_area > 0:
            rectangularity = area / box_area
            if rectangularity > 0.8:
                return {
                    'shape': 'rectangle',
                    'center': (int(rect[0][0]), int(rect[0][1])),
                    'size': (int(rect[1][0]), int(rect[1][1])),
                    'angle': rect[2],
                    'confidence': rectangularity
                }
        
        # Check for triangle
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(points.astype(np.float32), epsilon, True)
        
        if len(approx) == 3:
            return {
                'shape': 'triangle',
                'vertices': [tuple(pt[0]) for pt in approx],
                'confidence': 0.9
            }
        
        # Check for line (elongated shape)
        x, y, w, h = cv2.boundingRect(points.astype(np.float32))
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
        
        if aspect_ratio > 3 and circularity < 0.3:
            # Fit line
            [vx, vy, x0, y0] = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
            return {
                'shape': 'line',
                'vector': (float(vx), float(vy)),
                'point': (float(x0), float(y0)),
                'confidence': 0.85
            }
        
        return None
    
    def draw_perfect_shape(self, canvas, shape_info, color, thickness):
        """
        Draw perfect version of recognized shape.
        
        Args:
            canvas: Canvas to draw on
            shape_info: Shape information from recognize()
            color: Color for drawing
            thickness: Line thickness
        
        Returns:
            Modified canvas
        """
        if shape_info is None:
            return canvas
        
        shape_type = shape_info['shape']
        
        if shape_type == 'circle':
            center = shape_info['center']
            radius = shape_info['radius']
            cv2.circle(canvas, center, radius, color, thickness)
            
        elif shape_type == 'rectangle':
            center = shape_info['center']
            size = shape_info['size']
            angle = shape_info['angle']
            
            # Create rotated rectangle
            rect = (center, size, angle)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(canvas, [box], 0, color, thickness)
            
        elif shape_type == 'triangle':
            vertices = np.array(shape_info['vertices'], np.int32)
            vertices = vertices.reshape((-1, 1, 2))
            cv2.polylines(canvas, [vertices], True, color, thickness)
            
        elif shape_type == 'line':
            # Calculate line endpoints from vector and point
            vx, vy = shape_info['vector']
            x0, y0 = shape_info['point']
            
            # Project extreme points onto line
            pts = np.array([[x0, y0]])  # Simplified - would need actual stroke points
            
        return canvas


class RealTimeRecognizer:
    """Real-time character and shape recognition for air writing."""
    
    def __init__(self, char_model_path=None):
        self.char_recognizer = CharacterRecognizer(char_model_path)
        self.shape_recognizer = ShapeRecognizer()
        
        self.stroke_points = []
        self.is_drawing = False
        self.recognition_cooldown = 1.0  # seconds
        self.last_recognition_time = 0
        self.recognition_results = []
        
    def start_stroke(self):
        """Start new stroke."""
        self.stroke_points = []
        self.is_drawing = True
    
    def add_point(self, point):
        """Add point to current stroke."""
        if self.is_drawing:
            self.stroke_points.append(point)
    
    def end_stroke(self, canvas):
        """End stroke and recognize."""
        self.is_drawing = False
        
        current_time = time.time()
        if current_time - self.last_recognition_time < self.recognition_cooldown:
            return None
        
        # Try shape recognition first
        shape_info = self.shape_recognizer.recognize(self.stroke_points)
        
        if shape_info and shape_info.get('confidence', 0) > 0.75:
            result = {
                'type': 'shape',
                'data': shape_info,
                'timestamp': current_time
            }
        else:
            # Try character recognition
            char_result = self.char_recognizer.recognize_canvas(canvas)
            
            if char_result['success']:
                result = {
                    'type': 'character',
                    'data': char_result,
                    'timestamp': current_time
                }
            else:
                result = None
        
        if result:
            self.recognition_results.append(result)
            self.last_recognition_time = current_time
        
        return result
    
    def get_recognized_text(self):
        """Get concatenated recognized characters."""
        text = ''
        for result in self.recognition_results:
            if result['type'] == 'character':
                text += result['data']['character']
        return text
    
    def clear_history(self):
        """Clear recognition history."""
        self.recognition_results = []
        self.stroke_points = []


def create_training_data_from_canvas(canvas_samples, labels):
    """
    Create training data from canvas samples.
    
    Args:
        canvas_samples: List of canvas images
        labels: List of character labels
    
    Returns:
        x, y arrays for training
    """
    recognizer = CharacterRecognizer()
    x_data = []
    y_data = []
    
    for canvas, label in zip(canvas_samples, labels):
        char_region = recognizer.extract_character_region(canvas)
        if char_region is not None:
            processed = recognizer.preprocess_image(char_region)
            x_data.append(processed[0])
            
            # One-hot encode label
            label_idx = recognizer.classes.index(label) if label in recognizer.classes else -1
            if label_idx >= 0:
                y_encoded = np.zeros(len(recognizer.classes))
                y_encoded[label_idx] = 1
                y_data.append(y_encoded)
    
    return np.array(x_data), np.array(y_data)


# Example usage and testing
if __name__ == "__main__":
    print("AI Character Recognition Module")
    print("=" * 50)
    
    # Initialize recognizer
    recognizer = CharacterRecognizer()
    
    # Build and summarize model
    recognizer._build_model()
    recognizer.model.summary()
    
    print(f"\nSupported classes: {len(recognizer.classes)}")
    print(f"Classes: {', '.join(recognizer.classes[:20])}...")
    
    # Test with dummy data
    dummy_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    processed = recognizer.preprocess_image(dummy_image)
    print(f"\nProcessed image shape: {processed.shape}")
    
    # Test prediction (random weights, so results will be random)
    char, conf, alts = recognizer.predict(dummy_image)
    print(f"Prediction: {char} (confidence: {conf:.2f})")
