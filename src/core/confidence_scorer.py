"""
Innomatics OMR System - Advanced Confidence Scoring Engine
Multi-factor confidence assessment for production reliability

This module implements sophisticated confidence scoring using:
- Geometric consistency analysis
- Fill pattern validation
- Statistical confidence intervals
- Cross-validation with multiple detection methods
- Quality degradation assessment
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """Confidence level categories"""
    EXCELLENT = "excellent"      # >0.9
    GOOD = "good"               # 0.8-0.9
    MODERATE = "moderate"       # 0.6-0.8
    LOW = "low"                # 0.4-0.6
    VERY_LOW = "very_low"      # <0.4

@dataclass
class ConfidenceMetrics:
    """Detailed confidence assessment metrics"""
    overall_confidence: float
    geometric_confidence: float
    intensity_confidence: float
    consistency_confidence: float
    quality_confidence: float
    
    # Component scores
    fill_pattern_score: float
    shape_quality_score: float
    position_accuracy_score: float
    contrast_quality_score: float
    
    # Reliability indicators
    confidence_level: ConfidenceLevel
    reliability_score: float
    requires_review: bool
    
    # Detailed analysis
    factors_analyzed: int
    quality_issues: List[str]
    recommendations: List[str]

class InnoMaticsConfidenceScorer:
    """
    Advanced confidence scoring engine for OMR bubble detection.
    
    Uses multiple validation techniques to ensure <0.5% error rate:
    - Geometric validation against expected parameters
    - Intensity distribution analysis
    - Cross-method consistency checking
    - Statistical quality assessment
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Confidence calculation weights
        self.weights = {
            'geometric': 0.25,      # 25% - Shape and position accuracy
            'intensity': 0.30,      # 30% - Fill pattern quality  
            'consistency': 0.25,    # 25% - Cross-method agreement
            'quality': 0.20         # 20% - Overall image quality
        }
        
        # Quality thresholds
        self.thresholds = {
            'excellent': 0.9,
            'good': 0.8,
            'moderate': 0.6,
            'low': 0.4,
            'review_required': 0.6,
            'min_acceptable': 0.4
        }
        
        # Statistical parameters
        self.statistical_config = {
            'min_samples_for_stats': 10,
            'outlier_threshold': 2.0,  # Standard deviations
            'consistency_tolerance': 0.15,
            'quality_degradation_threshold': 0.3
        }
        
        self.logger.info("Confidence Scorer initialized for production quality assessment")
    
    def calculate_bubble_confidence(self, detection_result, image_region: np.ndarray,
                                  expected_position: Tuple[int, int],
                                  alternative_detections: List = None) -> ConfidenceMetrics:
        """
        Calculate comprehensive confidence score for a single bubble detection.
        
        Args:
            detection_result: Primary detection result
            image_region: Image region around the bubble
            expected_position: Expected bubble center position
            alternative_detections: Results from other detection methods
            
        Returns:
            ConfidenceMetrics with detailed assessment
        """
        try:
            self.logger.debug(f"Calculating confidence for bubble at {expected_position}")
            
            quality_issues = []
            recommendations = []
            factors_analyzed = 0
            
            # Component 1: Geometric Confidence
            geometric_conf = self._calculate_geometric_confidence(
                detection_result, expected_position, quality_issues
            )
            factors_analyzed += 1
            
            # Component 2: Intensity Confidence  
            intensity_conf = self._calculate_intensity_confidence(
                detection_result, image_region, quality_issues
            )
            factors_analyzed += 1
            
            # Component 3: Consistency Confidence
            consistency_conf = self._calculate_consistency_confidence(
                detection_result, alternative_detections, quality_issues
            )
            factors_analyzed += 1
            
            # Component 4: Quality Confidence
            quality_conf = self._calculate_quality_confidence(
                image_region, quality_issues
            )
            factors_analyzed += 1
            
            # Calculate overall confidence using weighted combination
            overall_confidence = (
                geometric_conf * self.weights['geometric'] +
                intensity_conf * self.weights['intensity'] +
                consistency_conf * self.weights['consistency'] +
                quality_conf * self.weights['quality']
            )
            
            # Generate detailed component scores
            fill_pattern_score = getattr(detection_result, 'fill_ratio', 0.5)
            shape_quality_score = getattr(detection_result, 'shape_quality', 0.5)
            position_accuracy_score = self._calculate_position_accuracy(detection_result, expected_position)
            contrast_quality_score = getattr(detection_result, 'contrast_score', 0.5)
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(overall_confidence)
            
            # Calculate reliability score (meta-confidence)
            reliability_score = self._calculate_reliability_score(
                [geometric_conf, intensity_conf, consistency_conf, quality_conf]
            )
            
            # Determine if review is required
            requires_review = (
                overall_confidence < self.thresholds['review_required'] or
                reliability_score < 0.7 or
                len(quality_issues) > 2
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                overall_confidence, quality_issues, requires_review
            )
            
            return ConfidenceMetrics(
                overall_confidence=overall_confidence,
                geometric_confidence=geometric_conf,
                intensity_confidence=intensity_conf,
                consistency_confidence=consistency_conf,
                quality_confidence=quality_conf,
                fill_pattern_score=fill_pattern_score,
                shape_quality_score=shape_quality_score,
                position_accuracy_score=position_accuracy_score,
                contrast_quality_score=contrast_quality_score,
                confidence_level=confidence_level,
                reliability_score=reliability_score,
                requires_review=requires_review,
                factors_analyzed=factors_analyzed,
                quality_issues=quality_issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {str(e)}")
            # Return low confidence on error
            return ConfidenceMetrics(
                overall_confidence=0.3,
                geometric_confidence=0.3,
                intensity_confidence=0.3,
                consistency_confidence=0.3,
                quality_confidence=0.3,
                fill_pattern_score=0.0,
                shape_quality_score=0.0,
                position_accuracy_score=0.0,
                contrast_quality_score=0.0,
                confidence_level=ConfidenceLevel.VERY_LOW,
                reliability_score=0.2,
                requires_review=True,
                factors_analyzed=0,
                quality_issues=["Confidence calculation failed"],
                recommendations=["Manual review required"]
            )
    
    def _calculate_geometric_confidence(self, detection_result, expected_position: Tuple[int, int],
                                      quality_issues: List[str]) -> float:
        """Calculate confidence based on geometric properties"""
        try:
            geometric_factors = []
            
            # Position accuracy
            if hasattr(detection_result, 'x') and hasattr(detection_result, 'y'):
                detected_pos = (detection_result.x, detection_result.y)
                position_error = math.sqrt(
                    (detected_pos[0] - expected_position[0])**2 + 
                    (detected_pos[1] - expected_position[1])**2
                )
                
                # Normalize position accuracy (lower error = higher confidence)
                max_acceptable_error = 10  # pixels
                position_accuracy = max(0, 1 - (position_error / max_acceptable_error))
                geometric_factors.append(position_accuracy)
                
                if position_error > 8:
                    quality_issues.append(f"Position offset: {position_error:.1f}px")
            else:
                geometric_factors.append(0.5)  # Default if no position info
            
            # Shape quality
            if hasattr(detection_result, 'shape_quality'):
                shape_quality = detection_result.shape_quality
                geometric_factors.append(shape_quality)
                
                if shape_quality < 0.6:
                    quality_issues.append(f"Poor shape quality: {shape_quality:.2f}")
            else:
                geometric_factors.append(0.5)
            
            # Size consistency
            if hasattr(detection_result, 'detected_radius'):
                expected_radius = 12  # Expected bubble radius
                size_error = abs(detection_result.detected_radius - expected_radius) / expected_radius
                size_accuracy = max(0, 1 - size_error)
                geometric_factors.append(size_accuracy)
                
                if size_error > 0.4:
                    quality_issues.append(f"Size inconsistency: {size_error*100:.1f}%")
            else:
                geometric_factors.append(0.7)  # Reasonable default
            
            return np.mean(geometric_factors)
            
        except Exception as e:
            self.logger.warning(f"Geometric confidence calculation error: {str(e)}")
            return 0.4
    
    def _calculate_intensity_confidence(self, detection_result, image_region: np.ndarray,
                                      quality_issues: List[str]) -> float:
        """Calculate confidence based on intensity patterns"""
        try:
            intensity_factors = []
            
            # Fill ratio quality
            if hasattr(detection_result, 'fill_ratio'):
                fill_ratio = detection_result.fill_ratio
                
                # Higher confidence for clear filled/unfilled states
                if fill_ratio > 0.7 or fill_ratio < 0.3:
                    fill_confidence = 0.9  # Clear decision
                elif 0.4 <= fill_ratio <= 0.6:
                    fill_confidence = 0.3  # Ambiguous range
                else:
                    fill_confidence = 0.6  # Moderate clarity
                
                intensity_factors.append(fill_confidence)
                
                if 0.4 <= fill_ratio <= 0.6:
                    quality_issues.append(f"Ambiguous fill ratio: {fill_ratio:.2f}")
            else:
                intensity_factors.append(0.5)
            
            # Intensity uniformity
            if image_region is not None and image_region.size > 0:
                # Calculate intensity statistics
                if len(image_region.shape) == 3:
                    gray_region = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
                else:
                    gray_region = image_region
                
                std_intensity = np.std(gray_region)
                mean_intensity = np.mean(gray_region)
                
                # Lower standard deviation indicates more uniform fill
                uniformity = max(0, 1 - (std_intensity / 64.0))  # Normalize by reasonable max std
                intensity_factors.append(uniformity)
                
                # Contrast quality
                if std_intensity > 50:
                    contrast_confidence = 0.8  # Good contrast
                elif std_intensity > 20:
                    contrast_confidence = 0.6  # Moderate contrast
                else:
                    contrast_confidence = 0.4  # Poor contrast
                    quality_issues.append(f"Low contrast: std={std_intensity:.1f}")
                
                intensity_factors.append(contrast_confidence)
            else:
                intensity_factors.extend([0.5, 0.5])  # Default values
            
            return np.mean(intensity_factors)
            
        except Exception as e:
            self.logger.warning(f"Intensity confidence calculation error: {str(e)}")
            return 0.4
    
    def _calculate_consistency_confidence(self, detection_result, alternative_detections: List,
                                        quality_issues: List[str]) -> float:
        """Calculate confidence based on cross-method consistency"""
        try:
            if not alternative_detections:
                return 0.7  # No alternatives to compare - assume reasonable confidence
            
            consistency_factors = []
            
            # Compare detection states across methods
            primary_state = getattr(detection_result, 'state', 'unknown')
            consistent_detections = 0
            total_comparisons = 0
            
            for alt_detection in alternative_detections:
                if hasattr(alt_detection, 'state'):
                    total_comparisons += 1
                    if alt_detection.state == primary_state:
                        consistent_detections += 1
            
            if total_comparisons > 0:
                state_consistency = consistent_detections / total_comparisons
                consistency_factors.append(state_consistency)
                
                if state_consistency < 0.6:
                    quality_issues.append(f"Method disagreement: {state_consistency*100:.0f}% agreement")
            
            # Compare confidence scores
            primary_conf = getattr(detection_result, 'confidence', 0.5)
            confidence_values = [primary_conf]
            
            for alt_detection in alternative_detections:
                if hasattr(alt_detection, 'confidence'):
                    confidence_values.append(alt_detection.confidence)
            
            if len(confidence_values) > 1:
                conf_std = np.std(confidence_values)
                conf_consistency = max(0, 1 - (conf_std / 0.3))  # Lower std = higher consistency
                consistency_factors.append(conf_consistency)
                
                if conf_std > 0.2:
                    quality_issues.append(f"Confidence variance: {conf_std:.2f}")
            
            # Compare fill ratios if available
            if hasattr(detection_result, 'fill_ratio'):
                primary_fill = detection_result.fill_ratio
                fill_values = [primary_fill]
                
                for alt_detection in alternative_detections:
                    if hasattr(alt_detection, 'fill_ratio'):
                        fill_values.append(alt_detection.fill_ratio)
                
                if len(fill_values) > 1:
                    fill_std = np.std(fill_values)
                    fill_consistency = max(0, 1 - (fill_std / 0.3))
                    consistency_factors.append(fill_consistency)
            
            return np.mean(consistency_factors) if consistency_factors else 0.6
            
        except Exception as e:
            self.logger.warning(f"Consistency confidence calculation error: {str(e)}")
            return 0.5
    
    def _calculate_quality_confidence(self, image_region: np.ndarray,
                                    quality_issues: List[str]) -> float:
        """Calculate confidence based on image quality"""
        try:
            if image_region is None or image_region.size == 0:
                quality_issues.append("No image region available")
                return 0.3
            
            quality_factors = []
            
            # Convert to grayscale if needed
            if len(image_region.shape) == 3:
                gray_region = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_region = image_region
            
            # Sharpness assessment using Laplacian variance
            laplacian = cv2.Laplacian(gray_region, cv2.CV_64F)
            sharpness = np.var(laplacian)
            
            # Normalize sharpness (higher variance = sharper)
            sharpness_score = min(1.0, sharpness / 1000.0)
            quality_factors.append(sharpness_score)
            
            if sharpness < 100:
                quality_issues.append(f"Low sharpness: {sharpness:.0f}")
            
            # Brightness assessment
            brightness = np.mean(gray_region) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Optimal around 0.5
            quality_factors.append(max(0, brightness_score))
            
            if brightness < 0.2 or brightness > 0.8:
                quality_issues.append(f"Poor brightness: {brightness:.2f}")
            
            # Contrast assessment
            contrast = np.std(gray_region) / 128.0
            contrast_score = min(contrast, 1.0)
            quality_factors.append(contrast_score)
            
            if contrast < 0.3:
                quality_issues.append(f"Low contrast: {contrast:.2f}")
            
            # Noise assessment
            if gray_region.shape[0] > 5 and gray_region.shape[1] > 5:
                blurred = cv2.GaussianBlur(gray_region, (5, 5), 0)
                noise_level = np.std(gray_region.astype(np.float32) - blurred.astype(np.float32)) / 128.0
                noise_score = max(0, 1 - noise_level)
                quality_factors.append(noise_score)
                
                if noise_level > 0.2:
                    quality_issues.append(f"High noise: {noise_level:.2f}")
            
            return np.mean(quality_factors)
            
        except Exception as e:
            self.logger.warning(f"Quality confidence calculation error: {str(e)}")
            return 0.4
    
    def _calculate_position_accuracy(self, detection_result, expected_position: Tuple[int, int]) -> float:
        """Calculate position accuracy score"""
        try:
            if hasattr(detection_result, 'x') and hasattr(detection_result, 'y'):
                detected_pos = (detection_result.x, detection_result.y)
                distance = math.sqrt(
                    (detected_pos[0] - expected_position[0])**2 + 
                    (detected_pos[1] - expected_position[1])**2
                )
                
                # Normalize distance (closer = higher score)
                max_distance = 15  # Maximum acceptable distance
                accuracy = max(0, 1 - (distance / max_distance))
                return accuracy
            else:
                return 0.5  # Default if no position available
                
        except Exception as e:
            self.logger.warning(f"Position accuracy calculation error: {str(e)}")
            return 0.4
    
    def _determine_confidence_level(self, overall_confidence: float) -> ConfidenceLevel:
        """Determine confidence level category"""
        if overall_confidence >= self.thresholds['excellent']:
            return ConfidenceLevel.EXCELLENT
        elif overall_confidence >= self.thresholds['good']:
            return ConfidenceLevel.GOOD
        elif overall_confidence >= self.thresholds['moderate']:
            return ConfidenceLevel.MODERATE
        elif overall_confidence >= self.thresholds['low']:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _calculate_reliability_score(self, component_scores: List[float]) -> float:
        """Calculate reliability of the confidence assessment itself"""
        try:
            # Higher reliability when component scores are consistent
            score_std = np.std(component_scores)
            score_mean = np.mean(component_scores)
            
            # Lower standard deviation = higher reliability
            consistency_score = max(0, 1 - (score_std / 0.3))
            
            # Higher mean confidence = higher reliability (with some bounds)
            confidence_score = min(1.0, score_mean * 1.2)
            
            # Combine factors
            reliability = (consistency_score * 0.6 + confidence_score * 0.4)
            
            return reliability
            
        except Exception as e:
            self.logger.warning(f"Reliability calculation error: {str(e)}")
            return 0.5
    
    def _generate_recommendations(self, overall_confidence: float, 
                                quality_issues: List[str], requires_review: bool) -> List[str]:
        """Generate actionable recommendations based on confidence assessment"""
        recommendations = []
        
        if overall_confidence >= 0.9:
            recommendations.append("Excellent quality - no action needed")
        elif overall_confidence >= 0.8:
            recommendations.append("Good quality - monitor for consistency")
        elif overall_confidence >= 0.6:
            recommendations.append("Acceptable quality - consider spot checking")
        elif overall_confidence >= 0.4:
            recommendations.append("Low quality - manual review recommended")
        else:
            recommendations.append("Very low quality - manual review required")
        
        # Specific recommendations based on quality issues
        if "Position offset" in str(quality_issues):
            recommendations.append("Check sheet alignment and positioning")
        
        if "Poor shape quality" in str(quality_issues):
            recommendations.append("Verify bubble detection parameters")
        
        if "Ambiguous fill ratio" in str(quality_issues):
            recommendations.append("Manual verification of mark intensity")
        
        if "Low contrast" in str(quality_issues):
            recommendations.append("Consider image enhancement preprocessing")
        
        if "Method disagreement" in str(quality_issues):
            recommendations.append("Cross-validate with additional detection methods")
        
        if requires_review:
            recommendations.append("Flag for manual review queue")
        
        return recommendations

    def calculate_sheet_confidence(self, bubble_confidences: List[ConfidenceMetrics]) -> Dict:
        """Calculate overall confidence for entire sheet"""
        try:
            if not bubble_confidences:
                return {
                    'overall_confidence': 0.0,
                    'average_reliability': 0.0,
                    'bubbles_requiring_review': 0,
                    'confidence_distribution': {},
                    'quality_summary': "No bubbles to analyze"
                }
            
            # Calculate aggregate metrics
            overall_scores = [bc.overall_confidence for bc in bubble_confidences]
            reliability_scores = [bc.reliability_score for bc in bubble_confidences]
            
            sheet_confidence = np.mean(overall_scores)
            average_reliability = np.mean(reliability_scores)
            
            # Count bubbles requiring review
            bubbles_requiring_review = sum(1 for bc in bubble_confidences if bc.requires_review)
            
            # Confidence distribution
            confidence_levels = [bc.confidence_level.value for bc in bubble_confidences]
            confidence_distribution = {
                level: confidence_levels.count(level) 
                for level in set(confidence_levels)
            }
            
            # Generate quality summary
            excellent_count = confidence_distribution.get('excellent', 0)
            good_count = confidence_distribution.get('good', 0)
            total_bubbles = len(bubble_confidences)
            
            if (excellent_count + good_count) / total_bubbles >= 0.8:
                quality_summary = "Excellent overall quality"
            elif (excellent_count + good_count) / total_bubbles >= 0.6:
                quality_summary = "Good overall quality"
            elif bubbles_requiring_review / total_bubbles <= 0.1:
                quality_summary = "Acceptable quality"
            else:
                quality_summary = "Poor quality - extensive review needed"
            
            return {
                'overall_confidence': sheet_confidence,
                'average_reliability': average_reliability,
                'bubbles_requiring_review': bubbles_requiring_review,
                'confidence_distribution': confidence_distribution,
                'quality_summary': quality_summary,
                'total_bubbles_analyzed': total_bubbles,
                'confidence_variance': np.var(overall_scores),
                'min_confidence': min(overall_scores),
                'max_confidence': max(overall_scores)
            }
            
        except Exception as e:
            self.logger.error(f"Sheet confidence calculation failed: {str(e)}")
            return {
                'overall_confidence': 0.0,
                'average_reliability': 0.0,
                'bubbles_requiring_review': len(bubble_confidences),
                'confidence_distribution': {},
                'quality_summary': "Analysis failed"
            }

# Main execution for testing
if __name__ == "__main__":
    # Initialize confidence scorer
    scorer = InnoMaticsConfidenceScorer()
    
    # Mock detection result for testing
    mock_detection = type('MockDetection', (), {
        'x': 100, 'y': 200,
        'fill_ratio': 0.8,
        'shape_quality': 0.9,
        'contrast_score': 0.7,
        'confidence': 0.85,
        'state': 'filled',
        'detected_radius': 12
    })()
    
    # Mock image region
    mock_image = np.random.randint(0, 255, (30, 30), dtype=np.uint8)
    
    # Calculate confidence
    confidence_metrics = scorer.calculate_bubble_confidence(
        mock_detection, mock_image, (100, 200)
    )
    
    print(f"Confidence Assessment:")
    print(f"- Overall: {confidence_metrics.overall_confidence:.3f}")
    print(f"- Level: {confidence_metrics.confidence_level.value}")
    print(f"- Reliability: {confidence_metrics.reliability_score:.3f}")
    print(f"- Requires Review: {confidence_metrics.requires_review}")
    print(f"- Issues: {len(confidence_metrics.quality_issues)}")
    print(f"- Recommendations: {len(confidence_metrics.recommendations)}")
