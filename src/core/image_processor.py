"""
Innomatics OMR System - Advanced Image Preprocessing Engine
Enterprise-grade mobile capture processing with shadow removal

This module handles the most challenging aspect of mobile OMR processing:
- Shadow detection and removal from mobile captures
- Illumination normalization for varying lighting conditions
- Perspective correction for camera angles
- Noise reduction and contrast enhancement
- Quality assessment and validation

Optimized for production deployment with <0.5% error rate target.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import logging
from pathlib import Path
from enum import Enum
import time

# --- MOCK CONFIGURATION FOR STANDALONE EXECUTION ---
# In a real-world scenario, this would be in a separate file.
class MockSettings:
    def __init__(self):
        self.IMAGE_PROCESSING = {
            'shadow_detection_method': 'hybrid',
            'illumination_correction_enabled': True,
            'perspective_detection_enabled': True,
            'corner_detection_quality': 0.01,
            'corner_detection_min_distance': 50,
            'shadow_dilation_kernel': 7,
            'homomorphic_cutoff': 30,
            'homomorphic_gamma_h': 2.0,
            'homomorphic_gamma_l': 0.3,
            'minimum_quality_score': 0.6,
            'excellent_quality_threshold': 0.85,
            'shadow_detection_threshold': 0.3,
            'enable_performance_monitoring': True,
            'cache_intermediate_results': False,
            'parallel_processing_enabled': True
        }
IMAGE_PROCESSING = MockSettings().IMAGE_PROCESSING
# --- END MOCK CONFIGURATION ---


logger = logging.getLogger(__name__)
# Set a more informative log format and level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# IMAGE PROCESSING DATA STRUCTURES
class ProcessingStage(Enum):
    """Image processing pipeline stages"""
    ORIGINAL = "original"
    SHADOW_REMOVED = "shadow_removed"
    ILLUMINATION_CORRECTED = "illumination_corrected"
    PERSPECTIVE_CORRECTED = "perspective_corrected"
    NOISE_REDUCED = "noise_reduced"
    CONTRAST_ENHANCED = "contrast_enhanced"
    FINAL_PROCESSED = "final_processed"

@dataclass
class ImageQualityMetrics:
    """Comprehensive image quality assessment"""
    overall_score: float                # 0-1 scale
    brightness_score: float
    contrast_score: float
    sharpness_score: float
    shadow_intensity: float
    noise_level: float
    perspective_distortion: float
    
    # Processing recommendations
    requires_shadow_removal: bool
    requires_contrast_enhancement: bool
    requires_noise_reduction: bool
    processing_difficulty: str          # easy, medium, hard, extreme

@dataclass
class ProcessingResult:
    """Complete processing result with metadata"""
    processed_image: np.ndarray
    processing_stages: Dict[str, np.ndarray]
    quality_metrics: ImageQualityMetrics
    processing_time: float
    success: bool
    warnings: List[str]
    parameters_used: Dict

class ShadowDetectionMethod(Enum):
    """Shadow detection algorithms"""
    MORPHOLOGICAL = "morphological"
    ILLUMINATION_INVARIANT = "illumination_invariant"
    BACKGROUND_SUBTRACTION = "background_subtraction"
    HYBRID = "hybrid"

# ENTERPRISE IMAGE PREPROCESSING ENGINE
class InnoMaticsImagePreprocessor:
    """
    Production-grade image preprocessing engine for mobile OMR captures.
    
    Features:
    - Advanced shadow detection and removal
    - Multi-stage illumination correction
    - Adaptive perspective correction
    - Real-time quality assessment
    - Batch processing optimization
    - Performance monitoring
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load configuration
        self.config = IMAGE_PROCESSING.copy()
        
        # Extended production configuration
        self.production_config = {
            **self.config,
            # Shadow removal parameters
            'shadow_detection_method': ShadowDetectionMethod.HYBRID,
            'shadow_threshold': 0.7,
            'shadow_dilation_kernel': 7,
            'shadow_gaussian_sigma': 20,
            
            # Illumination correction
            'illumination_correction_enabled': True,
            'homomorphic_cutoff': 30,
            'homomorphic_gamma_h': 2.0,
            'homomorphic_gamma_l': 0.3,
            
            # Perspective correction
            'perspective_detection_enabled': True,
            'corner_detection_quality': 0.01,
            'corner_detection_min_distance': 50,
            'perspective_threshold': 5.0,
            
            # Quality thresholds
            'minimum_quality_score': 0.6,
            'excellent_quality_threshold': 0.85,
            'shadow_detection_threshold': 0.3,
            
            # Performance settings
            'enable_performance_monitoring': True,
            'cache_intermediate_results': False, # Disable for memory efficiency
            'parallel_processing_enabled': True
        }
        
        # Initialize processing kernels
        self._initialize_morphological_kernels()
        
        # Performance tracking
        self.processing_stats = {
            'total_images_processed': 0,
            'average_processing_time': 0.0,
            'shadow_removal_success_rate': 0.0,
            'quality_improvement_average': 0.0
        }
        
        self.logger.info("Innomatics Image Preprocessor initialized for production")
    
    def _initialize_morphological_kernels(self):
        """Initialize morphological operation kernels"""
        kernel_size = self.production_config['shadow_dilation_kernel']
        self.kernels = {
            'shadow_detection': cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            ),
            'noise_reduction': cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            'opening': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            'closing': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        }

    # MAIN PROCESSING PIPELINE
    def process_image(self, image: np.ndarray, enable_stages: Optional[List[ProcessingStage]] = None) -> ProcessingResult:
        """
        Main image preprocessing pipeline for mobile OMR captures.
        
        Args:
            image: Input image (BGR or grayscale)
            enable_stages: List of processing stages to apply (None = all stages)
            
        Returns:
            ProcessingResult with processed image and metadata
        """
        start_time = time.time()
        
        # Default to all stages if none specified
        if enable_stages is None:
            enable_stages = list(ProcessingStage)
        
        try:
            self.logger.info("Starting comprehensive image preprocessing pipeline")
            
            # Initialize result tracking
            processing_stages = {ProcessingStage.ORIGINAL.value: image.copy()}
            warnings = []
            current_image = image.copy()
            
            # Initial quality assessment
            initial_quality = self._assess_image_quality(current_image)
            self.logger.info(f"Initial quality score: {initial_quality.overall_score:.3f}")
            
            # Stage 1: Shadow Detection and Removal
            if ProcessingStage.SHADOW_REMOVED in enable_stages:
                if initial_quality.requires_shadow_removal:
                    self.logger.info("Applying shadow removal - detected shadows")
                    current_image, shadow_warning = self._remove_shadows(current_image)
                    if shadow_warning:
                        warnings.append(shadow_warning)
                else:
                    self.logger.info("Skipping shadow removal - no significant shadows detected")
                processing_stages[ProcessingStage.SHADOW_REMOVED.value] = current_image.copy()
            
            # Stage 2: Illumination Correction
            if ProcessingStage.ILLUMINATION_CORRECTED in enable_stages:
                if self.production_config['illumination_correction_enabled']:
                    self.logger.info("Applying illumination correction")
                    current_image = self._correct_illumination(current_image)
                    processing_stages[ProcessingStage.ILLUMINATION_CORRECTED.value] = current_image.copy()
            
            # Stage 3: Perspective Correction
            if ProcessingStage.PERSPECTIVE_CORRECTED in enable_stages:
                if self.production_config['perspective_detection_enabled']:
                    current_image, perspective_applied = self._correct_perspective(current_image)
                    if not perspective_applied:
                        warnings.append("Perspective correction: Unable to detect sheet boundaries")
                processing_stages[ProcessingStage.PERSPECTIVE_CORRECTED.value] = current_image.copy()
            
            # Stage 4: Noise Reduction
            if ProcessingStage.NOISE_REDUCED in enable_stages:
                if initial_quality.requires_noise_reduction:
                    self.logger.info("Applying noise reduction")
                    current_image = self._reduce_noise(current_image)
                    processing_stages[ProcessingStage.NOISE_REDUCED.value] = current_image.copy()
            
            # Stage 5: Contrast Enhancement
            if ProcessingStage.CONTRAST_ENHANCED in enable_stages:
                if initial_quality.requires_contrast_enhancement:
                    self.logger.info("Applying contrast enhancement")
                    current_image = self._enhance_contrast(current_image)
                    processing_stages[ProcessingStage.CONTRAST_ENHANCED.value] = current_image.copy()
            
            # Final stage: Standardization
            if ProcessingStage.FINAL_PROCESSED in enable_stages:
                current_image = self._standardize_image(current_image)
                processing_stages[ProcessingStage.FINAL_PROCESSED.value] = current_image.copy()
            
            # Final quality assessment
            final_quality = self._assess_image_quality(current_image)
            processing_time = time.time() - start_time
            
            # Update performance statistics
            self._update_performance_stats(processing_time, initial_quality, final_quality)
            
            self.logger.info(f"Processing completed in {processing_time:.3f}s. "
                             f"Quality: {initial_quality.overall_score:.3f} -> {final_quality.overall_score:.3f}")
            
            return ProcessingResult(
                processed_image=current_image,
                processing_stages=processing_stages,
                quality_metrics=final_quality,
                processing_time=processing_time,
                success=True,
                warnings=warnings,
                parameters_used=self.production_config.copy()
            )
            
        except Exception as e:
            self.logger.error(f"Image processing failed: {str(e)}")
            return ProcessingResult(
                processed_image=image,  # Return original on failure
                processing_stages={ProcessingStage.ORIGINAL.value: image},
                quality_metrics=self._assess_image_quality(image),
                processing_time=time.time() - start_time,
                success=False,
                warnings=[f"Processing failed: {str(e)}"],
                parameters_used=self.production_config.copy()
            )

    # SHADOW DETECTION AND REMOVAL
    def _remove_shadows(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """
        Advanced shadow detection and removal using hybrid approach.
        Combines multiple techniques for robust shadow handling.
        """
        warning = None
        
        # Convert to grayscale for consistent processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Method selection based on image characteristics
        shadow_method = self.production_config['shadow_detection_method']
        
        if shadow_method == ShadowDetectionMethod.HYBRID:
            shadow_mask = self._detect_shadows_hybrid(gray)
        elif shadow_method == ShadowDetectionMethod.MORPHOLOGICAL:
            shadow_mask = self._detect_shadows_morphological(gray)
        elif shadow_method == ShadowDetectionMethod.ILLUMINATION_INVARIANT:
            shadow_mask = self._detect_shadows_illumination_invariant(image)
        else: # BACKGROUND_SUBTRACTION
            shadow_mask = self._detect_shadows_background_subtraction(gray)
        
        # Apply shadow removal if shadows detected
        if np.sum(shadow_mask) > 0:
            corrected_image = self._apply_shadow_correction(image, shadow_mask)
            
            # Validate correction quality
            shadow_coverage = np.sum(shadow_mask) / shadow_mask.size
            if shadow_coverage > 0.4: # More than 40% shadows
                warning = f"High shadow coverage detected ({shadow_coverage:.1%}), results may need review"
            
            return corrected_image, warning
        
        return image, None

    def _detect_shadows_hybrid(self, gray_image: np.ndarray) -> np.ndarray:
        """Hybrid shadow detection combining multiple methods"""
        # Method 1: Morphological shadow detection
        morph_shadows = self._detect_shadows_morphological(gray_image)
        
        # Method 2: Statistical shadow detection
        mean_intensity = np.mean(gray_image)
        std_intensity = np.std(gray_image)
        shadow_threshold = mean_intensity - 1.5 * std_intensity
        statistical_shadows = gray_image < shadow_threshold
        
        # Method 3: Gradient-based shadow detection
        gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
        gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_threshold = np.percentile(gradient_magnitude, 20)
        gradient_shadows = gradient_magnitude < gradient_threshold
        
        # Combine methods with weighted voting
        combined_shadows = (morph_shadows.astype(np.float32) * 0.4 +
                            statistical_shadows.astype(np.float32) * 0.4 +
                            gradient_shadows.astype(np.float32) * 0.2)
        
        # Apply threshold for final mask
        final_mask = (combined_shadows > 0.5)
        
        # Morphological operations to clean up the mask
        final_mask = cv2.morphologyEx(
            final_mask.astype(np.uint8), cv2.MORPH_OPEN, self.kernels['opening'])
        final_mask = cv2.morphologyEx(
            final_mask, cv2.MORPH_CLOSE, self.kernels['closing'])
        
        return final_mask.astype(bool)

    def _detect_shadows_morphological(self, image: np.ndarray) -> np.ndarray:
        """Morphological shadow detection method"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (21, 21), 0)
        
        # Morphological operations to detect shadow regions
        kernel = self.kernels['shadow_detection']
        
        # Top-hat transform to enhance bright regions
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        
        # Bottom-hat transform to enhance dark regions (shadows)
        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        
        # Combine transforms
        enhanced = cv2.add(image, tophat)
        enhanced = cv2.subtract(enhanced, blackhat)
        
        # Create shadow mask
        shadow_threshold = self.production_config['shadow_threshold'] * 255
        shadow_mask = (image < (enhanced - shadow_threshold))
        
        return shadow_mask

    def _detect_shadows_illumination_invariant(self, image: np.ndarray) -> np.ndarray:
        """Illumination-invariant shadow detection using R-G channel difference"""
        # This method is most effective on color images
        if len(image.shape) != 3:
            self.logger.warning("Illumination-invariant method requires a color image. Falling back to morphological.")
            return self._detect_shadows_morphological(image)
        
        # Convert to illumination-invariant space
        r, g, b = cv2.split(image.astype(np.float32))
        
        # Compute illumination-invariant image
        epsilon = 1e-6
        invariant = (r - g) / (r + g + epsilon)
        
        # Normalize to 0-255 range
        invariant = ((invariant + 1) * 127.5).astype(np.uint8)
        
        # Apply threshold to detect shadows
        _, shadow_mask = cv2.threshold(invariant, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return shadow_mask == 0 # Invert mask for shadows

    def _detect_shadows_background_subtraction(self, image: np.ndarray) -> np.ndarray:
        """Background subtraction-based shadow detection"""
        # Estimate background using morphological opening
        kernel_size = max(image.shape) // 20
        # Ensure odd kernel size for central pixel
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Smooth background
        background = cv2.GaussianBlur(background, (21, 21), 0)
        
        # Calculate difference
        diff = cv2.absdiff(image, background)
        
        # Create shadow mask (areas significantly darker than background)
        threshold = np.mean(diff) + np.std(diff)
        shadow_mask = (background - image) > threshold
        
        return shadow_mask

    def _apply_shadow_correction(self, image: np.ndarray, shadow_mask: np.ndarray) -> np.ndarray:
        """Apply shadow correction using adaptive histogram equalization"""
        corrected = image.copy()
        
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE only to shadow regions
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_corrected = clahe.apply(l)
            
            # Blend original and corrected luminance channels based on the mask
            l_final = np.where(shadow_mask, l_corrected, l)
            
            lab_corrected = cv2.merge([l_final, a, b])
            corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale correction
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            corrected_gray = clahe.apply(image)
            corrected = np.where(shadow_mask, corrected_gray, image)
            
        return corrected

    # ILLUMINATION CORRECTION
    def _correct_illumination(self, image: np.ndarray) -> np.ndarray:
        """Advanced illumination correction using homomorphic filtering"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0].astype(np.float32)
            
            l_corrected = self._apply_homomorphic_filter(l_channel)
            
            lab[:, :, 0] = np.clip(l_corrected, 0, 255).astype(np.uint8)
            corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            corrected = self._apply_homomorphic_filter(image.astype(np.float32))
            corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        return corrected

    def _apply_homomorphic_filter(self, channel: np.ndarray) -> np.ndarray:
        """Apply homomorphic filtering for illumination correction"""
        epsilon = 1e-6
        log_channel = np.log1p(channel - np.min(channel)) # Use log1p for robustness
        
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        
        cutoff = self.production_config['homomorphic_cutoff']
        gamma_h = self.production_config['homomorphic_gamma_h']
        gamma_l = self.production_config['homomorphic_gamma_l']
        
        u = np.arange(rows).reshape(-1, 1) - crow
        v = np.arange(cols).reshape(1, -1) - ccol
        
        duv = np.sqrt(u**2 + v**2)
        
        h_filter = (gamma_h - gamma_l) * (1 - np.exp(-(duv**2) / (2 * cutoff**2))) + gamma_l
        
        f_transform = np.fft.fft2(log_channel)
        f_transform = np.fft.fftshift(f_transform)
        
        filtered = h_filter * f_transform
        
        filtered = np.fft.ifftshift(filtered)
        result = np.fft.ifft2(filtered)
        result = np.real(result)
        
        result = np.expm1(result) + np.min(channel) # Use expm1
        
        return result

    # PERSPECTIVE CORRECTION
    def _correct_perspective(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Detect and correct perspective distortion"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=100,
                qualityLevel=self.production_config['corner_detection_quality'],
                minDistance=self.production_config['corner_detection_min_distance']
            )
            
            if corners is not None and len(corners) >= 4:
                corners = corners.reshape(-1, 2)
                
                # Sort corners to get a consistent ordering
                sorted_corners = self._sort_corners(corners)
                
                if sorted_corners is not None and len(sorted_corners) == 4:
                    height, width = image.shape[:2]
                    dst_corners = np.array([
                        [0, 0],
                        [width, 0], 
                        [width, height],
                        [0, height]
                    ], dtype=np.float32)
                    
                    transform_matrix = cv2.getPerspectiveTransform(sorted_corners, dst_corners)
                    
                    corrected = cv2.warpPerspective(image, transform_matrix, (width, height))
                    
                    # Recalculate perspective distortion after correction
                    perspective_distortion = self._assess_perspective_distortion(transform_matrix)
                    if perspective_distortion < self.production_config['perspective_threshold']:
                        return corrected, True
            
            return image, False
        
        except Exception as e:
            self.logger.warning(f"Perspective correction failed: {str(e)}")
            return image, False

    def _sort_corners(self, corners: np.ndarray) -> Optional[np.ndarray]:
        """Sort corners consistently as top-left, top-right, bottom-right, bottom-left"""
        if len(corners) < 4:
            return None
        
        # Calculate sum and difference for sorting
        sum_of_coords = corners.sum(axis=1)
        diff_of_coords = np.diff(corners, axis=1)
        
        sorted_corners = np.zeros_like(corners)
        sorted_corners[0] = corners[np.argmin(sum_of_coords)] # Top-left
        sorted_corners[2] = corners[np.argmax(sum_of_coords)] # Bottom-right
        sorted_corners[1] = corners[np.argmin(diff_of_coords)] # Top-right
        sorted_corners[3] = corners[np.argmax(diff_of_coords)] # Bottom-left
        
        return sorted_corners.astype(np.float32)

    def _assess_perspective_distortion(self, transform_matrix: np.ndarray) -> float:
        """Assess the degree of perspective distortion from the transformation matrix"""
        # A simple method: calculate the deviation from an affine matrix
        # This is a placeholder for a more complex, robust metric
        identity_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        diff = np.linalg.norm(transform_matrix - identity_matrix)
        return diff
    
    # NOISE REDUCTION AND ENHANCEMENT
    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction using bilateral filtering"""
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, self.kernels['noise_reduction'])
        return denoised

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced

    def _standardize_image(self, image: np.ndarray) -> np.ndarray:
        """Final standardization and normalization"""
        if len(image.shape) == 3:
            standardized = image.copy()
        else:
            standardized = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]])
        sharpened = cv2.filter2D(standardized, -1, kernel)
        
        standardized = cv2.addWeighted(standardized, 0.8, sharpened, 0.2, 0)
        
        return standardized

    # QUALITY ASSESSMENT
    def _assess_image_quality(self, image: np.ndarray) -> ImageQualityMetrics:
        """Comprehensive image quality assessment"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Brightness assessment
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2
            
            # Contrast assessment using standard deviation
            contrast = np.std(gray) / 128.0
            contrast_score = min(contrast, 1.0)
            
            # Sharpness assessment using Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = np.var(laplacian) / 10000.0
            sharpness_score = min(sharpness, 1.0)
            
            # Shadow intensity assessment
            shadow_threshold = np.percentile(gray, 25)
            shadow_pixels = np.sum(gray < shadow_threshold)
            shadow_intensity = shadow_pixels / gray.size
            
            # Noise level assessment
            noise_level = np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0)) / 128.0
            
            # Perspective distortion - simplified metric for this class
            perspective_distortion = 0.0 # Placeholder
            
            # Overall quality score (weighted combination)
            overall_score = (brightness_score * 0.2 + 
                             contrast_score * 0.3 + 
                             sharpness_score * 0.3 + 
                             (1 - shadow_intensity) * 0.2)
            
            requires_shadow_removal = shadow_intensity > self.production_config['shadow_detection_threshold']
            requires_contrast_enhancement = contrast_score < 0.5
            requires_noise_reduction = noise_level > 0.3
            
            if overall_score > 0.8:
                difficulty = "easy"
            elif overall_score > 0.6:
                difficulty = "medium"
            elif overall_score > 0.4:
                difficulty = "hard"
            else:
                difficulty = "extreme"
            
            return ImageQualityMetrics(
                overall_score=overall_score,
                brightness_score=brightness_score,
                contrast_score=contrast_score,
                sharpness_score=sharpness_score,
                shadow_intensity=shadow_intensity,
                noise_level=noise_level,
                perspective_distortion=perspective_distortion,
                requires_shadow_removal=requires_shadow_removal,
                requires_contrast_enhancement=requires_contrast_enhancement,
                requires_noise_reduction=requires_noise_reduction,
                processing_difficulty=difficulty
            )
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {str(e)}")
            return ImageQualityMetrics(
                overall_score=0.5,
                brightness_score=0.5,
                contrast_score=0.5,
                sharpness_score=0.5,
                shadow_intensity=0.0,
                noise_level=0.0,
                perspective_distortion=0.0,
                requires_shadow_removal=False,
                requires_contrast_enhancement=False,
                requires_noise_reduction=False,
                processing_difficulty="medium"
            )

    def _update_performance_stats(self, processing_time: float, 
                                 initial_quality: ImageQualityMetrics,
                                 final_quality: ImageQualityMetrics):
        """Update performance statistics"""
        self.processing_stats['total_images_processed'] += 1
        
        n = self.processing_stats['total_images_processed']
        current_avg = self.processing_stats['average_processing_time']
        self.processing_stats['average_processing_time'] = (current_avg * (n-1) + processing_time) / n
        
        quality_improvement = final_quality.overall_score - initial_quality.overall_score
        current_improvement = self.processing_stats['quality_improvement_average']
        self.processing_stats['quality_improvement_average'] = (current_improvement * (n-1) + quality_improvement) / n
        
        # New: Add shadow removal success rate tracking
        if initial_quality.requires_shadow_removal:
            success = 1 if final_quality.overall_score > initial_quality.overall_score else 0
            current_success_rate = self.processing_stats['shadow_removal_success_rate']
            self.processing_stats['shadow_removal_success_rate'] = (current_success_rate * (n-1) + success) / n

    # BATCH PROCESSING SUPPORT
    def process_batch(self, images: List[np.ndarray], 
                      enable_stages: Optional[List[ProcessingStage]] = None,
                      progress_callback=None) -> List[ProcessingResult]:
        """Process multiple images with progress tracking"""
        results = []
        total_images = len(images)
        
        for i, image in enumerate(images):
            result = self.process_image(image, enable_stages)
            results.append(result)
            
            if progress_callback:
                progress = (i + 1) / total_images
                progress_callback(progress, result)
        
        return results

    # UTILITY METHODS
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        return self.processing_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance tracking"""
        self.processing_stats = {
            'total_images_processed': 0,
            'average_processing_time': 0.0,
            'shadow_removal_success_rate': 0.0,
            'quality_improvement_average': 0.0
        }

# MAIN EXECUTION
if __name__ == "__main__":
    # To run this script, you'll need a sample image named `sample.jpg`
    # in a `data/samples` directory.
    # For example, you can create a dummy image with numpy for testing:
    # `dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)`
    # `cv2.imwrite("data/samples/sample.jpg", dummy_image)`
    
    # Initialize preprocessor
    preprocessor = InnoMaticsImagePreprocessor()
    
    # Test with sample image
    sample_image_path = Path("data/samples/sample.jpg")
    if sample_image_path.exists():
        sample_image = cv2.imread(str(sample_image_path))
        if sample_image is not None:
            result = preprocessor.process_image(sample_image)
            print("--- Processing Report ---")
            print(f"Success: {result.success}")
            print(f"Final Quality Score: {result.quality_metrics.overall_score:.3f}")
            print(f"Processing Time: {result.processing_time:.3f}s")
            if result.warnings:
                print("Warnings:")
                for warning in result.warnings:
                    print(f"- {warning}")
        else:
            print(f"Error: Could not read image at {sample_image_path}")
    else:
        print(f"Error: Sample image not found at {sample_image_path}. Please create one to test the code.")
