#!/usr/bin/env python3
"""
Innomatics OMR - Interactive Coordinate Calibration
Get exact bubble positions from your OMR image
"""

import cv2
import numpy as np
import json
from pathlib import Path

class OMRCoordinateCalibrator:
    """Interactive tool to calibrate bubble coordinates"""
    
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.original_image = self.image.copy()
        self.clicked_points = []
        self.current_question = 1
        self.current_option = 'A'
        self.coordinates = {}
        
        # Create window
        cv2.namedWindow('OMR Calibration', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('OMR Calibration', 1200, 900)
        cv2.setMouseCallback('OMR Calibration', self.mouse_callback)
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for coordinate selection"""
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add clicked point
            self.clicked_points.append((x, y))
            
            # Store coordinate
            if self.current_question not in self.coordinates:
                self.coordinates[self.current_question] = {}
            
            self.coordinates[self.current_question][self.current_option] = (x, y)
            
            # Draw circle at clicked point
            cv2.circle(self.image, (x, y), 8, (0, 255, 0), 2)
            
            # Add label
            label = f"Q{self.current_question}{self.current_option}"
            cv2.putText(self.image, label, (x-15, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            print(f"✅ Q{self.current_question}{self.current_option}: ({x}, {y})")
            
            # Move to next option/question
            self.advance_position()
            
            # Update display
            self.update_display()
    
    def advance_position(self):
        """Move to next option or question"""
        
        if self.current_option == 'A':
            self.current_option = 'B'
        elif self.current_option == 'B':
            self.current_option = 'C'
        elif self.current_option == 'C':
            self.current_option = 'D'
        elif self.current_option == 'D':
            self.current_option = 'A'
            self.current_question += 1
            
            # Stop at question 21 (first column complete)
            if self.current_question > 20:
                print("🎉 First column completed! Press 's' to save or continue clicking...")
    
    def update_display(self):
        """Update the display with current status"""
        
        display_image = self.image.copy()
        
        # Add instructions
        instructions = [
            f"Current: Q{self.current_question}{self.current_option}",
            "Click on bubble positions",
            "Press 's' to save coordinates",
            "Press 'r' to reset", 
            "Press 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(display_image, instruction, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show progress
        progress = f"Progress: {len(self.coordinates)}/100 questions"
        cv2.putText(display_image, progress, (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow('OMR Calibration', display_image)
    
    def run_calibration(self):
        """Run the interactive calibration process"""
        
        print("🎯 OMR COORDINATE CALIBRATION")
        print("=" * 40)
        print("Instructions:")
        print("1. Click on each bubble position in order (A, B, C, D for each question)")
        print("2. Start with Question 1 Option A (top-left of first column)")
        print("3. Continue through Q1A, Q1B, Q1C, Q1D, then Q2A, Q2B, etc.")
        print("4. Press 's' to save coordinates when done")
        print("5. Press 'q' to quit")
        print()
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                break
            elif key == ord('s'):  # Save
                self.save_coordinates()
            elif key == ord('r'):  # Reset
                self.reset_calibration()
            elif key == ord('a'):  # Auto-generate based on pattern
                self.auto_generate_coordinates()
        
        cv2.destroyAllWindows()
    
    def reset_calibration(self):
        """Reset the calibration"""
        
        self.image = self.original_image.copy()
        self.clicked_points = []
        self.coordinates = {}
        self.current_question = 1
        self.current_option = 'A'
        self.update_display()
        print("🔄 Calibration reset")
    
    def auto_generate_coordinates(self):
        """Auto-generate coordinates based on existing pattern"""
        
        if len(self.coordinates) < 2:
            print("❌ Need at least 2 questions calibrated to auto-generate")
            return
        
        print("🤖 Auto-generating coordinates based on pattern...")
        
        # Analyze pattern from existing coordinates
        if 1 in self.coordinates and 2 in self.coordinates:
            q1_coords = self.coordinates[1]
            q2_coords = self.coordinates[2]
            
            # Calculate row spacing
            row_spacing = q2_coords['A'][1] - q1_coords['A'][1]
            
            # Calculate option spacing
            option_spacing = q1_coords['B'][0] - q1_coords['A'][0]
            
            print(f"   Detected row spacing: {row_spacing}")
            print(f"   Detected option spacing: {option_spacing}")
            
            # Generate coordinates for remaining questions in first column
            base_x = q1_coords['A'][0]
            base_y = q1_coords['A'][1]
            
            for q in range(1, 21):  # First 20 questions
                if q not in self.coordinates:
                    self.coordinates[q] = {}
                
                question_y = base_y + (q - 1) * row_spacing
                
                for i, option in enumerate(['A', 'B', 'C', 'D']):
                    option_x = base_x + i * option_spacing
                    self.coordinates[q][option] = (option_x, question_y)
                    
                    # Draw on image
                    cv2.circle(self.image, (option_x, question_y), 6, (255, 0, 0), 2)
            
            # Generate for other columns (estimate column spacing)
            if len(self.clicked_points) >= 8:  # Have some points from multiple questions
                col_spacing = 140  # Initial estimate
                
                for col in range(1, 5):  # Columns 2-5
                    col_base_x = base_x + col * col_spacing
                    
                    for q in range(21 + col*20 - 20, 21 + col*20):  # Questions for this column
                        if q <= 100:
                            if q not in self.coordinates:
                                self.coordinates[q] = {}
                            
                            question_y = base_y + ((q-1) % 20) * row_spacing
                            
                            for i, option in enumerate(['A', 'B', 'C', 'D']):
                                option_x = col_base_x + i * option_spacing
                                self.coordinates[q][option] = (option_x, question_y)
                                
                                # Draw on image
                                cv2.circle(self.image, (option_x, question_y), 4, (0, 0, 255), 1)
            
            self.update_display()
            print(f"✅ Auto-generated coordinates for {len(self.coordinates)} questions")
    
    def save_coordinates(self):
        """Save the calibrated coordinates"""
        
        if not self.coordinates:
            print("❌ No coordinates to save")
            return
        
        # Save to multiple locations
        locations = [
            'calibrated_coordinates.json',
            'src/core/calibrated_coordinates.json',
            'config/calibrated_coordinates.json'
        ]
        
        for location in locations:
            Path(location).parent.mkdir(parents=True, exist_ok=True)
            
            with open(location, 'w') as f:
                json.dump(self.coordinates, f, indent=2)
            
            print(f"💾 Coordinates saved to: {location}")
        
        print(f"✅ Saved {len(self.coordinates)} question coordinates")
        
        # Also save debug image
        cv2.imwrite('calibrated_coordinates_debug.jpg', self.image)
        print("💾 Debug image saved: calibrated_coordinates_debug.jpg")

def quick_auto_calibrate():
    """Quick automatic calibration based on your OMR layout"""
    
    print("🚀 QUICK AUTO-CALIBRATION FOR YOUR OMR FORMAT")
    print("=" * 50)
    
    # Load your image
    image_path = "data/samples/Img1.jpeg"
    image = cv2.imread(image_path)
    
    if image is None:
        print("❌ Could not load image")
        return
    
    height, width = image.shape[:2]
    print(f"📐 Image dimensions: {width} x {height}")
    
    # REFINED coordinates based on your actual OMR layout analysis
    coordinates = {}
    
    # Layout analysis from your improved_bubble_detection.jpg
    # These coordinates should match your actual bubble positions
    
    # Column positions (based on visual analysis)
    columns = [
        {'start_x': 170, 'spacing': 18, 'name': 'Python'},      # Q1-20
        {'start_x': 310, 'spacing': 18, 'name': 'Data Analysis'}, # Q21-40
        {'start_x': 450, 'spacing': 18, 'name': 'MySQL'},       # Q41-60
        {'start_x': 590, 'spacing': 18, 'name': 'Power BI'},    # Q61-80
        {'start_x': 730, 'spacing': 18, 'name': 'Adv Stats'}    # Q81-100
    ]
    
    # Row specifications
    start_y = 255  # First question row
    row_spacing = 23  # Spacing between rows
    
    question_num = 1
    
    for col_idx, col_info in enumerate(columns):
        print(f"   Generating {col_info['name']} coordinates...")
        
        for row in range(20):  # 20 questions per column
            if question_num > 100:
                break
            
            y_pos = start_y + row * row_spacing
            
            coordinates[question_num] = {
                'A': (col_info['start_x'], y_pos),
                'B': (col_info['start_x'] + col_info['spacing'], y_pos),
                'C': (col_info['start_x'] + col_info['spacing'] * 2, y_pos),
                'D': (col_info['start_x'] + col_info['spacing'] * 3, y_pos)
            }
            
            question_num += 1
    
    # Test the coordinates with actual bubble detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    debug_image = image.copy()
    filled_answers = {}
    
    for q_num, q_coords in coordinates.items():
        best_option = None
        max_fill_ratio = 0
        
        for option, (x, y) in q_coords.items():
            # Extract bubble region
            radius = 12
            y1, y2 = max(0, y-radius), min(gray.shape[0], y+radius)
            x1, x2 = max(0, x-radius), min(gray.shape[1], x+radius)
            
            if y2 > y1 and x2 > x1:
                bubble_region = gray[y1:y2, x1:x2]
                
                if bubble_region.size > 0:
                    # Analyze fill ratio
                    dark_pixels = np.sum(bubble_region < 128)
                    total_pixels = bubble_region.size
                    fill_ratio = dark_pixels / total_pixels
                    
                    # Check if this is the most filled option for this question
                    if fill_ratio > 0.4 and fill_ratio > max_fill_ratio:
                        max_fill_ratio = fill_ratio
                        best_option = option
            
            # Draw all bubble positions (small circles)
            cv2.circle(debug_image, (x, y), 4, (100, 100, 100), 1)
        
        # Mark the best answer (if any)
        if best_option and max_fill_ratio > 0.4:
            filled_answers[q_num] = {'option': best_option, 'fill_ratio': max_fill_ratio}
            x, y = q_coords[best_option]
            cv2.circle(debug_image, (x, y), 8, (0, 255, 0), 2)
            cv2.putText(debug_image, f"Q{q_num}:{best_option}", (x-15, y-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
    # Save coordinates
    locations = [
        'quick_calibrated_coordinates.json',
        'src/core/quick_calibrated_coordinates.json'
    ]
    
    for location in locations:
        Path(location).parent.mkdir(parents=True, exist_ok=True)
        with open(location, 'w') as f:
            json.dump(coordinates, f, indent=2)
        print(f"💾 Quick coordinates saved to: {location}")
    
    # Save debug image
    cv2.imwrite('quick_calibrated_debug.jpg', debug_image)
    print("💾 Quick debug image saved: quick_calibrated_debug.jpg")
    
    print(f"\n🎯 QUICK CALIBRATION RESULTS:")
    print(f"✅ Generated coordinates for {len(coordinates)} questions")
    print(f"✅ Found filled answers for {len(filled_answers)} questions")
    
    if len(filled_answers) > 50:
        print("🎉 EXCELLENT! Quick calibration found many answers!")
    elif len(filled_answers) > 20:
        print("👍 GOOD! Quick calibration is working well")
    else:
        print("⚠️  Few answers found - coordinates may need fine-tuning")
    
    return coordinates, filled_answers

def main():
    """Main function"""
    
    print("🎯 OMR COORDINATE CALIBRATION TOOL")
    print("=" * 40)
    
    choice = input("Choose calibration method:\n1. Interactive (click on bubbles)\n2. Quick auto-calibration\nEnter 1 or 2: ").strip()
    
    if choice == "1":
        # Interactive calibration
        try:
            calibrator = OMRCoordinateCalibrator("data/samples/Img1.jpeg")
            calibrator.run_calibration()
        except Exception as e:
            print(f"❌ Error: {e}")
    
    elif choice == "2":
        # Quick auto-calibration
        coordinates, answers = quick_auto_calibrate()
        
        if len(answers) > 30:
            print(f"\n🎉 SUCCESS! Quick calibration works well.")
            print(f"Found {len(answers)} filled answers - ready for production!")
            
            # Update the main OMR processor coordinates
            try:
                import shutil
                shutil.copy('quick_calibrated_coordinates.json', 'src/core/precise_coordinates.json')
                print("✅ Updated main OMR processor with new coordinates")
            except:
                print("ℹ️  Manually copy quick_calibrated_coordinates.json to update the system")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
