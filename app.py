import streamlit as st
# Set page config must be the first Streamlit command
# st.set_page_config(page_title="🎓 Mentor Demo Review System", layout="wide")

import torch
import os
import numpy as np
import librosa
import whisper
from openai import OpenAI
import tempfile
import warnings
import re
from contextlib import contextmanager
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import subprocess
import json
import shutil
from pathlib import Path
import time
from faster_whisper import WhisperModel
import soundfile as sf
import logging
from typing import Optional, Dict, Any, List, Tuple
import sys
import multiprocessing
import concurrent.futures
import hashlib
import threading
import random
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle, TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from io import BytesIO
import plotly.express as px
import psutil
import GPUtil
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import pickle
import io
import plotly.graph_objects as go
from datetime import datetime

# Filter out ScriptRunContext warnings
warnings.filterwarnings('ignore', '.*ScriptRunContext!.*')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === Google Sheets Integration ===
SHEET_ID = "1SXeGOOT8D4wEN0n0CTQWbW0Q7xOSOawBQvFXxKM8hIs"  # <-- Set your Google Sheet ID here
SHEET_NAME = "Report"  # Tab name as provided

METRIC_THRESHOLDS = {
    'monotone_score': {'excellent': 0.1, 'good': 0.3, 'acceptable': 0.5},
    'pitch_variation': {'excellent': 25, 'good': 20, 'acceptable': 15, 'min': 10},
    'direction_changes': {'min': 200, 'optimal_min': 300, 'optimal_max': 600, 'max': 800},
    'fillers_per_min': {'excellent': 1, 'good': 2, 'acceptable': 3, 'max_acceptable': 4},
    'errors_per_min': {'excellent': 0.2, 'good': 0.5, 'acceptable': 1, 'max_acceptable': 1.5},
    'words_per_minute': {'optimal_min': 130, 'optimal_max': 150, 'good_min': 120, 'good_max': 180, 'acceptable_min': 110, 'acceptable_max': 190, 'max_acceptable_min': 100, 'max_acceptable_max': 200}
}

# Balanced metric thresholds with more realistic expectations
# Philosophy: Balance teaching content and communication skills appropriately
# Communication and teaching skills are equally weighted to ensure comprehensive evaluation
TEACHING_METRIC_WEIGHTS = {
    'content_accuracy': 0.30,      # Strong subject matter knowledge is critical
    'industry_examples': 0.20,     # Practical application is important  
    'qna_accuracy': 0.20,         # Question handling shows expertise
    'engagement': 0.15,           # Student interaction matters
    'communication': 0.15         # Communication is equally important as teaching
}

# More realistic assessment thresholds aligned with new scoring
TEACHING_ASSESSMENT_THRESHOLDS = {
    'excellent': 8.5,   # 85% or above (8.5/10)
    'good': 7.0,       # 70% or above (7.0/10) 
    'acceptable': 5.5,  # 55% or above (5.5/10)
    'needs_improvement': 4.0,  # 40% or above (4.0/10)
    'poor': 0.0        # Below 40%
}

def append_metrics_to_sheet(evaluation_data, filename, sheet_id=SHEET_ID, sheet_name=SHEET_NAME):
    """Append all PDF report metrics as a row to the Google Sheet. If the sheet is empty, add the header first."""
    try:
        credentials_json = os.getenv('GOOGLE_DRIVE_CREDENTIALS')
        if not credentials_json:
            logger.error("GOOGLE_DRIVE_CREDENTIALS environment variable not found")
            return
        credentials_info = json.loads(credentials_json)
        creds = service_account.Credentials.from_service_account_info(
            credentials_info,
            scopes=[
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive',
            ]
        )
        service = build('sheets', 'v4', credentials=creds)
        sheet = service.spreadsheets()

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        speech_metrics = evaluation_data.get("speech_metrics", {})
        teaching_data = evaluation_data.get("teaching", {})
        audio_features = evaluation_data.get("audio_features", {})
        recommendations = evaluation_data.get("recommendations", {})

        # --- Recommendations & Summary ---
        hiring_rec = recommendations.get("hiringRecommendation", {})
        hiring_score = hiring_rec.get("score", "")
        hiring_reasons = ", ".join(hiring_rec.get("reasons", []))
        hiring_strengths = ", ".join(hiring_rec.get("strengths", []))
        hiring_concerns = ", ".join(hiring_rec.get("concerns", []))
        summary = recommendations.get("summary", "")
        geography_fit = recommendations.get("geographyFit", "")
        rigor = recommendations.get("rigor", "")
        profile_matches = recommendations.get("profileMatches", [])
        profile_matches_str = "; ".join([
            f"{p.get('profile','')}: {'Yes' if p.get('match') else 'No'} ({p.get('reason','')})" for p in profile_matches
        ])

        # --- Key Teaching Metrics ---
        concept_data = teaching_data.get("Concept Assessment", {})
        content_accuracy = concept_data.get("Subject Matter Accuracy", {})
        content_score = content_accuracy.get("Score", "")
        examples = concept_data.get("Examples and Business Context", {})
        examples_score = examples.get("Score", "")
        speed_data = speech_metrics.get("speed", {})
        words_per_minute = speed_data.get("wpm", "")
        pace_score = 1 if 120 <= float(words_per_minute or 0) <= 180 else 0
        qna_data = concept_data.get("Question Handling", {})
        qna_details = qna_data.get("Details", {})
        response_accuracy = qna_details.get("ResponseAccuracy", {}).get("Score", "")
        response_completeness = qna_details.get("ResponseCompleteness", {}).get("Score", "")
        accent_info = audio_features.get("accent_classification", {})
        accent = accent_info.get("accent", "")
        accent_confidence = accent_info.get("confidence", "")
        accent_score = 1 if float(accent_confidence or 0) > 0.7 else 0

        # --- Confidence Assessment ---
        fluency_data = speech_metrics.get("fluency", {})
        fillers_per_min = fluency_data.get("fillersPerMin", "")
        errors_per_min = fluency_data.get("errorsPerMin", "")
        intonation_data = speech_metrics.get("intonation", {})
        pitch_variation = intonation_data.get("pitchVariation", "")
        pitch_mean = intonation_data.get("pitch", 0) or 1
        pitch_var_percent = float(pitch_variation or 0) / float(pitch_mean or 1) * 100 if pitch_mean else 0
        filler_confidence = "High" if float(fillers_per_min or 0) <= 2 else "Medium" if float(fillers_per_min or 0) <= 4 else "Low"
        error_confidence = "High" if float(errors_per_min or 0) <= 0.5 else "Medium" if float(errors_per_min or 0) <= 1 else "Low"
        pitch_confidence = "High" if 20 <= pitch_var_percent <= 40 else "Low"
        confidence_score = sum([
            1 if filler_confidence == "High" else 0.5 if filler_confidence == "Medium" else 0,
            1 if error_confidence == "High" else 0.5 if error_confidence == "Medium" else 0,
            1 if pitch_confidence == "High" else 0
        ]) / 3 * 100
        confidence_assessment = "Very Confident" if confidence_score >= 80 else "Moderately Confident" if confidence_score >= 60 else "Shows Nervousness"

        # --- Communication Metrics ---
        total_words = speed_data.get("total_words", "")
        duration_minutes = speed_data.get("duration_minutes", "")
        detected_fillers = ", ".join([f["word"]+":"+str(f["count"]) for f in fluency_data.get("detectedFillers", [])])
        detected_errors = ", ".join([f'{e.get("type","")}({e.get("count","")}):{e.get("context","")}' for e in fluency_data.get("detectedErrors", [])])
        flow_data = speech_metrics.get("flow", {})
        pauses_per_min = flow_data.get("pausesPerMin", "")
        intonation_display_data = audio_features.get("monotone_score", ""), audio_features.get("pitch_mean", ""), audio_features.get("pitch_variation_coeff", ""), audio_features.get("direction_changes_per_min", "")
        energy_display_data = audio_features.get("mean_amplitude", ""), audio_features.get("amplitude_deviation", "")

        # Calculate acceptance status for communication metrics
        speed_accepted = 1 if 120 <= float(words_per_minute or 0) <= 180 else 0
        fillers_accepted = 1 if float(fillers_per_min or 0) <= 3 else 0
        errors_accepted = 1 if float(errors_per_min or 0) <= 1 else 0
        pitch_accepted = 1 if float(pitch_variation or 0) >= 20 else 0

        # --- Teaching Analysis (flatten all categories) ---
        concept_flat = {}
        for cat, details in concept_data.items():
            if isinstance(details, dict):
                concept_flat[f"Concept_{cat}_Score"] = details.get("Score", "")
                concept_flat[f"Concept_{cat}_Citations"] = ", ".join(details.get("Citations", []))
                # If there are nested details (e.g., for QnA)
                if "Details" in details and isinstance(details["Details"], dict):
                    for subcat, subdetails in details["Details"].items():
                        concept_flat[f"Concept_{cat}_{subcat}_Score"] = subdetails.get("Score", "")
                        concept_flat[f"Concept_{cat}_{subcat}_Citations"] = ", ".join(subdetails.get("Citations", []))
        code_flat = {}
        code_data = teaching_data.get("Code Assessment", {})
        for cat, details in code_data.items():
            if isinstance(details, dict):
                code_flat[f"Code_{cat}_Score"] = details.get("Score", "")
                code_flat[f"Code_{cat}_Citations"] = ", ".join(details.get("Citations", []))

        # --- Build row and header ---
        row = [
            now, filename,
            hiring_score, hiring_reasons, hiring_strengths, hiring_concerns, summary, geography_fit, rigor, profile_matches_str,
            content_score, examples_score, words_per_minute, pace_score, response_accuracy, response_completeness, accent, accent_confidence, accent_score,
            fillers_per_min, errors_per_min, pitch_var_percent, filler_confidence, error_confidence, pitch_confidence, confidence_score, confidence_assessment,
            total_words, duration_minutes, detected_fillers, detected_errors, pauses_per_min,
            intonation_display_data[0], intonation_display_data[1], intonation_display_data[2], intonation_display_data[3],
            energy_display_data[0], energy_display_data[1],
            speed_accepted, fillers_accepted, errors_accepted, pitch_accepted,  # Add acceptance status columns
        ]
        # Add all concept/code assessment fields
        for k in sorted(concept_flat):
            row.append(concept_flat[k])
        for k in sorted(code_flat):
            row.append(code_flat[k])

        header = [
            "Timestamp", "Filename",
            "Hiring Score", "Hiring Reasons", "Hiring Strengths", "Hiring Concerns", "Summary", "Geography Fit", "Teaching Rigor", "Profile Matches",
            "Content Accuracy Score", "Industry Examples Score", "Teaching Pace (WPM)", "Pace Score", "QnA Response Accuracy", "QnA Response Completeness", "Accent", "Accent Confidence", "Accent Score",
            "Fillers/Min", "Errors/Min", "Pitch Variation %", "Filler Confidence", "Error Confidence", "Pitch Confidence", "Overall Confidence Score", "Confidence Assessment",
            "Total Words", "Duration (min)", "Detected Fillers", "Detected Errors", "Pauses/Min",
            "Monotone Score", "Pitch Mean (Hz)", "Pitch Variation Coeff (%)", "Direction Changes/Min",
            "Mean Amplitude", "Amplitude Deviation",
            "Speed Accepted", "Fillers Accepted", "Errors Accepted", "Pitch Accepted",  # Add acceptance status headers
        ]
        for k in sorted(concept_flat):
            header.append(k)
        for k in sorted(code_flat):
            header.append(k)

        # --- Check if sheet is empty (no data) ---
        result = sheet.values().get(
            spreadsheetId=sheet_id,
            range=f"{sheet_name}!A1:A2"
        ).execute()
        values = result.get('values', [])
        is_empty = not values or all(len(row) == 0 for row in values)

        # If empty, write header first
        if is_empty:
            sheet.values().append(
                spreadsheetId=sheet_id,
                range=f"{sheet_name}!A1",
                valueInputOption="USER_ENTERED",
                insertDataOption="INSERT_ROWS",
                body={"values": [header]}
            ).execute()
            logger.info(f"Header row written to Google Sheet {sheet_id} - {sheet_name}")

        # Append row
        sheet.values().append(
            spreadsheetId=sheet_id,
            range=f"{sheet_name}!A1",
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body={"values": [row]}
        ).execute()
        logger.info(f"Appended all metrics for {filename} to Google Sheet.")
    except Exception as e:
        logger.error(f"Failed to append metrics to Google Sheet: {e}")

def clear_gpu_memory():
    """Clear GPU memory and cache"""
    try:
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Clear any remaining GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        
        logger.info("GPU memory cleared successfully")
    except Exception as e:
        logger.warning(f"Error clearing GPU memory: {e}")

def monitor_memory_usage():
    """Monitor system memory and GPU usage"""
    try:
        # Get system memory info
        memory = psutil.virtual_memory()
        logger.info(f"System Memory: {memory.percent}% used ({memory.used / 1024 / 1024 / 1024:.1f}GB / {memory.total / 1024 / 1024 / 1024:.1f}GB)")
        
        # Get GPU info if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu = GPUtil.getGPUs()[i]
                logger.info(f"GPU {i} Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
    except Exception as e:
        logger.warning(f"Error monitoring memory usage: {e}")

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors"""
    pass

@contextmanager
def temporary_file(suffix: Optional[str] = None):
    """Context manager for temporary file handling"""
    temp_path = tempfile.mktemp(suffix=suffix)
    try:
        yield temp_path
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_path}: {e}")

class ProgressTracker:
    """Tracks progress across multiple processing steps"""
    def __init__(self, status_container, progress_bar):
        self.status = status_container
        self.progress = progress_bar
        self.current_step = 0
        self.total_steps = 5  # Total number of main processing steps
        self.substep_container = st.empty()  # Add container for substep details
        self.metrics_container = st.container()  # Add container for metrics
        
    def update(self, progress: float, message: str, substep: str = "", metrics: Dict[str, Any] = None):
        """Update progress bar and status message with enhanced UI feedback
        
        Args:
            progress: Progress within current step (0-1)
            message: Main status message
            substep: Optional substep detail
            metrics: Optional dictionary of metrics to display
        """
        # Calculate overall progress (each step is 20% of total)
        overall_progress = min((self.current_step + progress) / self.total_steps, 1.0)
        
        # Update progress bar with smoother animation
        self.progress.progress(overall_progress)
        
        # Update main status with color coding
        status_html = f"""
        <div class="status-message {'status-processing' if overall_progress < 1 else 'status-complete'}">
            <h4>{message}</h4>
        """
        if substep:
            status_html += f"<p>{substep}</p>"
        status_html += "</div>"
        
        self.status.markdown(status_html, unsafe_allow_html=True)
        
        # Display metrics if provided
        if metrics:
            with self.metrics_container:
                cols = st.columns(len(metrics))
                for col, (metric_name, metric_value) in zip(cols, metrics.items()):
                    with col:
                        st.metric(
                            label=metric_name,
                            value=metric_value if isinstance(metric_value, (int, float)) else str(metric_value)
                        )
    
    def next_step(self):
        """Move to next processing step with visual feedback"""
        self.current_step = min(self.current_step + 1, self.total_steps)
        
        # Clear substep container for new step
        self.substep_container.empty()
        
        # Update progress with completion animation
        if self.current_step == self.total_steps:
            self.progress.progress(1.0)
            self.status.markdown("""
                <div class="status-message status-complete">
                    <h4>✅ Processing Complete!</h4>
                </div>
            """, unsafe_allow_html=True)
        

    def error(self, message: str):
        """Display error message with visual feedback"""
        self.status.markdown(f"""
            <div class="status-message status-error">
                <h4>❌ Error</h4>
                <p>{message}</p>
            </div>
        """, unsafe_allow_html=True)
        

class AudioFeatureExtractor:
    """Handles audio feature extraction with improved pause detection"""
    def __init__(self):
        self.sr = 16000
        self.hop_length = 512
        self.n_fft = 2048
        self.chunk_duration = 300
        # Parameters for pause detection
        self.min_pause_duration = 4  # minimum pause duration in seconds
        self.silence_threshold = -40    # dB threshold for silence
        
        # Initialize accent classifier
        try:
            from speechbrain.pretrained import EncoderClassifier
            self.accent_classifier = EncoderClassifier.from_hparams(
                source="Jzuluaga/accent-id-commonaccent_ecapa",
                savedir="pretrained_models/accent-id-commonaccent_ecapa"
            )
            self.has_accent_classifier = True
        except Exception as e:
            logger.warning(f"Could not initialize accent classifier: {e}")
            self.has_accent_classifier = False
    def _analyze_pauses(self, silent_frames, frame_time):
        """Analyze pauses with minimal memory usage."""
        pause_durations = []
        current_pause = 0

        for is_silent in silent_frames:
            if is_silent:
                current_pause += 1
            elif current_pause > 0:
                duration = current_pause * frame_time
                if duration > 0.5:  # Only count pauses longer than 300ms
                    pause_durations.append(duration)
                current_pause = 0

        if pause_durations:
            return {
                'total_pauses': len(pause_durations),
                'mean_pause_duration': float(np.mean(pause_durations))
            }
        return {
            'total_pauses': 0,
            'mean_pause_duration': 0.0
        }

    def _trim_audio_to_duration(self, audio_path: str, max_duration_seconds: int = 300) -> str:
        """Trim audio file to specified duration (default 5 minutes) and return path to trimmed file"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sr)
            
            # Calculate samples for max duration
            max_samples = sr * max_duration_seconds
            
            # Trim audio if longer than max duration
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            
            # Create temporary file that will persist until explicitly deleted
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_file.name
            temp_file.close()  # Close the file handle but keep the file
            
            # Write the audio data
            sf.write(temp_path, audio, sr)
            return temp_path
                
        except Exception as e:
            logger.error(f"Error trimming audio: {e}")
            return audio_path

    def classify_accent(self, audio_path: str) -> Dict[str, Any]:
        """Classify the accent from the first 5 minutes of the audio file"""
        if not self.has_accent_classifier:
            return {"accent": "Unknown", "confidence": 0.0}
        
        temp_path = None
        try:
            # Trim audio to 5 minutes for accent detection
            temp_path = self._trim_audio_to_duration(audio_path, max_duration_seconds=300)
            
            # Classify accent using trimmed audio
            out_prob, score, index, text_lab = self.accent_classifier.classify_file(temp_path)
            
            # Convert torch tensors to Python types and create result
            result = {
                "accent": text_lab[0] if isinstance(text_lab, list) else str(text_lab),
                "confidence": float(score[0]) if hasattr(score, '__len__') else float(score),
                "probabilities": {
                    str(i): float(prob) 
                    for i, prob in enumerate(out_prob[0])
                } if hasattr(out_prob, '__len__') else {},
                "note": "Analysis based on first 5 minutes of audio"
            }
            
            # Log the classification results for debugging
            logger.info(f"Accent classification results - Label: {text_lab}, Score: {score}, Index: {index}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in accent classification: {e}")
            return {"accent": "Unknown", "confidence": 0.0}
            
        finally:
            # Clean up temporary file
            if temp_path and temp_path != audio_path:
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"Error cleaning up temporary file: {e}")

    def extract_features(self, audio_path: str, progress_callback=None) -> Dict[str, float]:
        try:
            if progress_callback:
                progress_callback(0.1, "Loading audio file...")
            
            # Load audio with proper sample rate
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Calculate amplitude features
            rms = librosa.feature.rms(y=audio)[0]
            mean_amplitude = float(np.mean(rms)) * 100  # Scale for better readability
            
            # Enhanced pitch analysis for monotone detection
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                sr=sr,
                fmin=70,
                fmax=400,
                frame_length=2048
            )
            
            # Filter out zero and NaN values
            valid_f0 = f0[np.logical_and(voiced_flag == 1, ~np.isnan(f0))]
            
            # Calculate pitch statistics for monotone detection
            pitch_mean = float(np.mean(valid_f0)) if len(valid_f0) > 0 else 0
            pitch_std = float(np.std(valid_f0)) if len(valid_f0) > 0 else 0
            pitch_range = float(np.ptp(valid_f0)) if len(valid_f0) > 0 else 0  # Peak-to-peak range
            
            # Calculate pitch variation coefficient (normalized standard deviation)
            pitch_variation_coeff = (pitch_std / pitch_mean * 100) if pitch_mean > 0 else 0
            
            # Calculate monotone score based on multiple factors
            # 1. Low pitch variation (monotone speakers have less variation)
            variation_factor = min(1.0, max(0.0, 1.0 - (pitch_variation_coeff / 30.0)))
            
            # 2. Small pitch range relative to mean pitch (monotone speakers have smaller ranges)
            range_ratio = (pitch_range / pitch_mean * 100) if pitch_mean > 0 else 0
            range_factor = min(1.0, max(0.0, 1.0 - (range_ratio / 100.0)))
            
            # 3. Few pitch direction changes (monotone speakers have fewer changes)
            pitch_changes = np.diff(valid_f0) if len(valid_f0) > 1 else np.array([])
            direction_changes = np.sum(np.diff(np.signbit(pitch_changes))) if len(pitch_changes) > 0 else 0
            changes_per_minute = direction_changes / (len(audio) / sr / 60) if len(audio) > 0 else 0
            changes_factor = min(1.0, max(0.0, 1.0 - (changes_per_minute / 300.0)))
            
            # Calculate final monotone score (0-1, higher means more monotonous)
            monotone_score = (variation_factor * 0.4 + range_factor * 0.3 + changes_factor * 0.3)
            
            # Log the factors for debugging
            logger.info(f"""Monotone score calculation:
                Pitch variation coeff: {pitch_variation_coeff:.2f}
                Variation factor: {variation_factor:.2f}
                Range ratio: {range_ratio:.2f}
                Range factor: {range_factor:.2f}
                Changes per minute: {changes_per_minute:.2f}
                Changes factor: {changes_factor:.2f}
                Final monotone score: {monotone_score:.2f}
            """)
            
            # Calculate pauses per minute
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            silence_frames = rms_db < self.silence_threshold
            frame_time = self.hop_length / sr
            pause_analysis = self._analyze_pauses(silence_frames, frame_time)
            
            # Calculate pauses per minute
            duration_minutes = len(audio) / sr / 60
            pauses_per_minute = float(pause_analysis['total_pauses'] / duration_minutes if duration_minutes > 0 else 0)
            
            # Add accent classification
            if progress_callback:
                progress_callback(0.9, "Classifying accent...")
            
            accent_info = self.classify_accent(audio_path)
            
            return {
                "pitch_mean": pitch_mean,
                "pitch_std": pitch_std,
                "pitch_range": pitch_range,
                "pitch_variation_coeff": pitch_variation_coeff,
                "monotone_score": monotone_score,  # Added monotone score to output
                "mean_amplitude": mean_amplitude,
                "amplitude_deviation": float(np.std(rms) / np.mean(rms)) if np.mean(rms) > 0 else 0,
                "pauses_per_minute": pauses_per_minute,
                "duration": float(len(audio) / sr),
                "rising_patterns": int(np.sum(np.diff(valid_f0) > 0)) if len(valid_f0) > 1 else 0,
                "falling_patterns": int(np.sum(np.diff(valid_f0) < 0)) if len(valid_f0) > 1 else 0,
                "variations_per_minute": float(len(valid_f0) / (len(audio) / sr / 60)) if len(audio) > 0 else 0,
                "direction_changes_per_min": changes_per_minute,
                "accent_classification": accent_info
            }
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            raise AudioProcessingError(f"Feature extraction failed: {str(e)}")


    def _process_chunk(self, chunk: np.ndarray) -> Dict[str, Any]:
        """Process a single chunk of audio with improved pause detection"""
        # Calculate STFT
        D = librosa.stft(chunk, n_fft=self.n_fft, hop_length=self.hop_length)
        S = np.abs(D)
        
        # Calculate RMS energy in dB
        rms = librosa.feature.rms(S=S)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Detect pauses using silence threshold
        is_silence = rms_db < self.silence_threshold
        frame_time = self.hop_length / self.sr
        pause_analysis = self._analyze_pauses(is_silence, frame_time)
        
        # Calculate pitch features
        f0, voiced_flag, _ = librosa.pyin(
            chunk,
            sr=self.sr,
            fmin=70,
            fmax=400,
            frame_length=self.n_fft
        )
        
        return {
            "rms": rms,
            "f0": f0[voiced_flag == 1] if f0 is not None else np.array([]),
            "duration": len(chunk) / self.sr,
            "pause_count": pause_analysis['total_pauses'],
            "mean_pause_duration": pause_analysis['mean_pause_duration']
        }

    def _combine_features(self, features: List[Dict[str, Any]]) -> Dict[str, float]:
        """Combine features from multiple chunks"""
        all_f0 = np.concatenate([f["f0"] for f in features if len(f["f0"]) > 0])
        all_rms = np.concatenate([f["rms"] for f in features])
        
        pitch_mean = np.mean(all_f0) if len(all_f0) > 0 else 0
        pitch_std = np.std(all_f0) if len(all_f0) > 0 else 0
        
        return {
            "pitch_mean": float(pitch_mean),
            "pitch_std": float(pitch_std),
            "mean_amplitude": float(np.mean(all_rms)),
            "amplitude_deviation": float(np.std(all_rms) / np.mean(all_rms)) if np.mean(all_rms) > 0 else 0,
            "rising_patterns": int(np.sum(np.diff(all_f0) > 0)) if len(all_f0) > 1 else 0,
            "falling_patterns": int(np.sum(np.diff(all_f0) < 0)) if len(all_f0) > 1 else 0,
            "variations_per_minute": float((np.sum(np.diff(all_f0) != 0) if len(all_f0) > 1 else 0) / 
                                        (sum(f["duration"] for f in features) / 60))
        }

class ContentAnalyzer:
    """Analyzes teaching content using OpenAI API"""
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.retry_count = 3
        self.retry_delay = 1
        
    def analyze_content(self, transcript: str, progress_callback=None) -> Dict[str, Any]:
        """Analyze teaching content with strict validation and robust JSON handling"""
        default_structure = {
            "Concept Assessment": {
                "Subject Matter Accuracy": {
                    "Score": 0,
                    "Citations": ["[00:00] Unable to assess - insufficient evidence"]
                },
                "First Principles Approach": {
                    "Score": 0,
                    "Citations": ["[00:00] Unable to assess - insufficient evidence"]
                },
                "Examples and Business Context": {
                    "Score": 0,
                    "Citations": ["[00:00] Unable to assess - insufficient evidence"]
                },
                "Cohesive Storytelling": {
                    "Score": 0,
                    "Citations": ["[00:00] Unable to assess - insufficient evidence"]
                },
                "Engagement and Interaction": {
                    "Score": 0,
                    "Citations": ["[00:00] Unable to assess - insufficient evidence"]
                },
                "Professional Tone": {
                    "Score": 0,
                    "Citations": ["[00:00] Unable to assess - insufficient evidence"]
                },
                "Question Handling": {
                    "Score": 0,
                    "Citations": ["[00:00] Unable to assess - insufficient evidence"],
                    "Details": {
                        "ResponseAccuracy": {
                            "Score": 0,
                            "Citations": ["[00:00] Unable to assess - insufficient evidence"],
                            "Requirements": {
                                "TechnicalAccuracy": False,
                                "FactualCorrectness": False,
                                "NoMisleadingInfo": False
                            }
                        },
                        "ResponseCompleteness": {
                            "Score": 0,
                            "Citations": ["[00:00] Unable to assess - insufficient evidence"],
                            "Requirements": {
                                "AddressesAllParts": False,
                                "ProvidesContext": False,
                                "IncludesExamples": False
                            }
                        },
                        "ConfidenceLevel": {
                            "Score": 0,
                            "Citations": ["[00:00] Unable to assess - insufficient evidence"],
                            "Requirements": {
                                "ClearDelivery": False,
                                "NoHesitation": False,
                                "HandlesFollowUp": False
                            }
                        },
                        "ResponseTime": {
                            "Score": 0,
                            "Citations": ["[00:00] Unable to assess - insufficient evidence"],
                            "AverageResponseTime": 0.0
                        },
                        "ClarificationSkills": {
                            "Score": 0,
                            "Citations": ["[00:00] Unable to assess - insufficient evidence"],
                            "Requirements": {
                                "AsksProbing": False,
                                "ConfirmsUnderstanding": False,
                                "ReframesComplex": False
                            }
                        }
                    }
                }
            },
            "Code Assessment": {
                "Depth of Explanation": {
                    "Score": 0,
                    "Citations": ["[00:00] Unable to assess - insufficient evidence"]
                },
                "Output Interpretation": {
                    "Score": 0,
                    "Citations": ["[00:00] Unable to assess - insufficient evidence"]
                },
                "Breaking down Complexity": {
                    "Score": 0,
                    "Citations": ["[00:00] Unable to assess - insufficient evidence"]
                }
            }
        }

        for attempt in range(self.retry_count):
            try:
                if progress_callback:
                    progress_callback(0.2, "Preparing content analysis...")
                
                prompt = self._create_analysis_prompt(transcript)
                
                if progress_callback:
                    progress_callback(0.5, "Processing with AI model...")
                
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",  # Using GPT-4 for better analysis
                        messages=[
                            {"role": "system", "content": """You are a strict teaching evaluator focusing on core teaching competencies.
                             For each assessment point, you MUST include specific timestamps [MM:SS] from the transcript.
                             Never use [00:00] as a placeholder - only use actual timestamps from the transcript.
                             Each citation must include both the timestamp and a relevant quote showing evidence.
                             
                             Score of 1 requires meeting ALL criteria below with clear evidence.
                             Score of 0 if ANY major teaching deficiency is present.
                             
                             Citations format: "[MM:SS] Exact quote from transcript showing evidence"
                             
                             Maintain high standards and require clear evidence of quality teaching."""},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3
                    )
                    
                    logger.info("API call successful")
                except Exception as api_error:
                    logger.error(f"API call failed: {str(api_error)}")
                    raise
                
                result_text = response.choices[0].message.content.strip()
                # Remove markdown code block markers if present
                result_text = result_text.replace('```json\n', '').replace('\n```', '')
                logger.info(f"Raw API response: {result_text[:500]}...")
                
                try:
                    # Parse the API response
                    result = json.loads(result_text)
                    
                    # Validate and clean up the structure
                    for category in ["Concept Assessment", "Code Assessment"]:
                        if category not in result:
                            result[category] = default_structure[category]
                        else:
                            for subcategory in default_structure[category]:
                                if subcategory not in result[category]:
                                    result[category][subcategory] = default_structure[category][subcategory]
                                else:
                                    # Ensure proper structure and non-empty citations
                                    entry = result[category][subcategory]
                                    if not isinstance(entry, dict):
                                        entry = {"Score": 0, "Citations": []}
                                    if "Score" not in entry:
                                        entry["Score"] = 0
                                    if "Citations" not in entry or not entry["Citations"]:
                                        entry["Citations"] = [f"[{self._get_timestamp(transcript)}] Insufficient evidence for assessment"]
                                    # Ensure Score is either 0 or 1
                                    entry["Score"] = 1 if entry["Score"] == 1 else 0
                                    result[category][subcategory] = entry
                    
                    return result
                    
                except json.JSONDecodeError as json_error:
                    logger.error(f"JSON parsing error: {json_error}")
                    if attempt == self.retry_count - 1:
                        # On final attempt, try to extract structured data
                        return self._extract_structured_data(result_text)
                    
            except Exception as e:
                logger.error(f"Content analysis attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.retry_count - 1:
                    return default_structure
                time.sleep(self.retry_delay * (2 ** attempt))
        
        return default_structure

    def _get_timestamp(self, transcript: str) -> str:
        """Generate a reasonable timestamp based on transcript length"""
        # Calculate approximate time based on word count
        words = len(transcript.split())
        minutes = words // 150  # Assuming 150 words per minute
        seconds = (words % 150) * 60 // 150
        return f"{minutes:02d}:{seconds:02d}"

    def _extract_structured_data(self, text: str) -> Dict[str, Any]:
        """Extract structured data from text response when JSON parsing fails"""
        default_structure = {
            "Concept Assessment": {},
            "Code Assessment": {}
        }
        
        try:
            # Simple pattern matching to extract scores and citations
            sections = text.split('\n\n')
            current_category = None
            current_subcategory = None
            
            for section in sections:
                if "Concept Assessment" in section:
                    current_category = "Concept Assessment"
                elif "Code Assessment" in section:
                    current_category = "Code Assessment"
                elif current_category and ':' in section:
                    title, content = section.split(':', 1)
                    current_subcategory = title.strip()
                    
                    # Extract score (assuming 0 or 1 is mentioned)
                    score = 1 if "pass" in content.lower() or "score: 1" in content.lower() else 0
                    
                    # Extract citations (assuming they're in [MM:SS] format)
                    citations = re.findall(r'\[\d{2}:\d{2}\].*?(?=\[|$)', content)
                    citations = [c.strip() for c in citations if c.strip()]
                    
                    if not citations:
                        citations = ["No specific citations found"]
                    
                    if current_category and current_subcategory:
                        if current_category not in default_structure:
                            default_structure[current_category] = {}
                        default_structure[current_category][current_subcategory] = {
                            "Score": score,
                            "Citations": citations
                        }
            
            return default_structure
        except Exception as e:
            logger.error(f"Error extracting structured data: {e}")
            return default_structure

    def _create_analysis_prompt(self, transcript: str) -> str:
        """Create the analysis prompt with stricter evaluation criteria"""
        # First try to extract existing timestamps
        timestamps = re.findall(r'\[(\d{2}:\d{2})\]', transcript)
        
        if timestamps:
            timestamp_instruction = f"""Use the EXACT timestamps from the transcript (e.g. {', '.join(timestamps[:3])}).
Do not create new timestamps."""
        else:
            # Calculate approximate timestamps based on word position
            timestamp_instruction = """Generate timestamps based on word position:
1. Count words from start of transcript
2. Calculate time: (word_count / 150) minutes
3. Format as [MM:SS]"""

        prompt_template = """Analyze this teaching content with balanced standards. Each criterion should be evaluated fairly, avoiding both excessive strictness and leniency.

Score 1 if MOST key requirements are met with clear evidence. Score 0 if MULTIPLE significant requirements are not met.
You MUST provide specific citations with timestamps [MM:SS] for each assessment point.

Transcript:
{transcript}

Timestamp Instructions:
{timestamp_instruction}

Required JSON response format:
{{
    "Concept Assessment": {{
        "Subject Matter Accuracy": {{
            "Score": 0 or 1,
            "Citations": ["[MM:SS] Exact quote showing evidence"]
        }},
        "First Principles Approach": {{
            "Score": 0 or 1,
            "Citations": ["[MM:SS] Exact quote showing evidence"]
        }},
        "Examples and Business Context": {{
            "Score": 0 or 1,
            "Citations": ["[MM:SS] Exact quote showing evidence"]
        }},
        "Cohesive Storytelling": {{
            "Score": 0 or 1,
            "Citations": ["[MM:SS] Exact quote showing evidence"]
        }},
        "Engagement and Interaction": {{
            "Score": 0 or 1,
            "Citations": ["[MM:SS] Exact quote showing evidence of question handling"],
            "QuestionConfidence": {{
                "Score": 0 or 1,
                "Citations": ["[MM:SS] Exact quote showing evidence of question handling"]
            }}
        }},
        "Professional Tone": {{
            "Score": 0 or 1,
            "Citations": ["[MM:SS] Exact quote showing evidence"]
        }},
        "Question Handling": {{
            "Score": 0 or 1,
            "Citations": ["[MM:SS] Exact quote showing evidence of question handling"],
            "Details": {{
                "ResponseAccuracy": {{
                    "Score": 0 or 1,
                    "Citations": ["[MM:SS] Exact quote showing evidence of response accuracy"]
                }},
                "ResponseCompleteness": {{
                    "Score": 0 or 1,
                    "Citations": ["[MM:SS] Exact quote showing evidence of response completeness"]
                }},
                "ConfidenceLevel": {{
                    "Score": 0 or 1,
                    "Citations": ["[MM:SS] Exact quote showing evidence of confidence level"]
                }}
            }}
        }}
    }},
    "Code Assessment": {{
        "Depth of Explanation": {{
            "Score": 0 or 1,
            "Citations": ["[MM:SS] Exact quote showing evidence"]
        }},
        "Output Interpretation": {{
            "Score": 0 or 1,
            "Citations": ["[MM:SS] Exact quote showing evidence"]
        }},
        "Breaking down Complexity": {{
            "Score": 0 or 1,
            "Citations": ["[MM:SS] Exact quote showing evidence"]
        }}
    }}
}}

Balanced Scoring Criteria:

Subject Matter Accuracy:
✓ Score 1 if MOST:
- Shows good technical knowledge
- Uses appropriate terminology
- Explains concepts correctly
✗ Score 0 if MULTIPLE:
- Contains significant technical errors
- Uses consistently incorrect terminology
- Misrepresents core concepts

First Principles Approach:
✓ Score 1 if MOST:
- Introduces fundamental concepts
- Shows logical progression
- Connects related concepts
✗ Score 0 if MULTIPLE:
- Skips essential fundamentals
- Shows unclear progression
- Fails to connect concepts

Examples and Business Context:
✓ Score 1 if MOST:
- Provides relevant examples
- Shows business application
- Demonstrates practical value
✗ Score 0 if MULTIPLE:
- Lacks meaningful examples
- Missing practical context
- Examples don't aid learning

Cohesive Storytelling:
✓ Score 1 if MOST:
- Shows clear structure
- Has logical transitions
- Maintains consistent theme
✗ Score 0 if MULTIPLE:
- Has unclear structure
- Shows jarring transitions
- Lacks coherent theme

Engagement and Interaction:
✓ Score 1 if MOST:
- Shows good audience interaction
- Encourages participation
- Answers questions confidently and accurately
- Maintains engagement throughout
✗ Score 0 if MULTIPLE:
- Limited interaction
- Ignores audience
- Shows uncertainty in answers
- Fails to maintain engagement

Question Confidence Scoring:
✓ Score 1 if MOST:
- Provides clear, direct answers
- Shows deep understanding
- Handles follow-ups well
- Maintains composure
✗ Score 0 if MULTIPLE:
- Shows uncertainty
- Provides unclear answers
- Struggles with follow-ups
- Shows nervousness

Professional Tone:
✓ Score 1 if MOST:
- Uses appropriate language
- Shows confidence
- Maintains clarity
✗ Score 0 if MULTIPLE:
- Uses inappropriate language
- Shows consistent uncertainty
- Is frequently unclear

Depth of Explanation:
✓ Score 1 if MOST:
- Explains core concepts
- Covers key details
- Discusses implementation
✗ Score 0 if MULTIPLE:
- Misses core concepts
- Skips important details
- Lacks implementation depth

Output Interpretation:
✓ Score 1 if MOST:
- Explains key results
- Covers common errors
- Discusses performance
✗ Score 0 if MULTIPLE:
- Unclear about results
- Ignores error cases
- Misses performance aspects

Breaking down Complexity:
✓ Score 1 if MOST:
- Breaks down concepts
- Shows clear steps
- Builds understanding
✗ Score 0 if MULTIPLE:
- Keeps concepts too complex
- Skips important steps
- Creates confusion

Important:
- Each citation must include timestamp and relevant quote
- Score 1 requires meeting MOST (not all) criteria
- Score 0 requires MULTIPLE significant issues
- Use specific evidence from transcript
- Balance between being overly strict and too lenient

Question Handling Assessment Criteria (ALL must be met for score of 1):

1. Response Accuracy (Must meet ALL):
   - Technical information must be 100% accurate
   - All factual statements must be verifiable
   - No misleading or ambiguous information
   - Citations must show clear evidence of accurate responses

2. Response Completeness (Must meet ALL):
   - Must address ALL parts of each question
   - Must provide necessary context
   - Must include relevant examples where appropriate
   - No partial or incomplete answers accepted

3. Confidence Level (Must meet ALL):
   - Clear, authoritative delivery
   - No hesitation or uncertainty in responses
   - Confident handling of follow-up questions
   - Maintains professional tone throughout

4. Response Time:
   - Must respond within 3-5 seconds of question
   - Longer response times must be justified by question complexity
   - Must acknowledge question immediately even if full response needs time

5. Clarification Skills (Must meet ALL):
   - Asks probing questions when needed
   - Confirms understanding before answering
   - Reframes complex questions effectively
   - Ensures question intent is fully understood

Score 0 if ANY of the following are present:
- Any technical inaccuracy
- Incomplete or partial answers
- Excessive hesitation or uncertainty
- Failure to ask clarifying questions when needed
- Missing examples or context
- Delayed responses without justification
"""

        return prompt_template.format(
            transcript=transcript,
            timestamp_instruction=timestamp_instruction
        )

    def _evaluate_speech_metrics(self, transcript: str, audio_features: Dict[str, float], 
                           progress_callback=None) -> Dict[str, Any]:
        try:
            if progress_callback:
                progress_callback(0.2, "Calculating speech metrics...")

            # Calculate words and duration
            words = len(transcript.split())
            duration_minutes = float(audio_features.get('duration', 0)) / 60
            
            # Calculate words per minute
            words_per_minute = float(words / duration_minutes if duration_minutes > 0 else 0)
            
            # Use LLM to detect fillers and errors
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": """Analyze the speech transcript for:
                        1. Filler words (um, uh, like, you know, etc.)
                        2. Speech errors (repeated words, incomplete sentences, grammatical mistakes)
                        
                        Return a JSON with:
                        {
                            "fillers": [{"word": "filler_word", "count": number}],
                            "errors": [{"type": "error_type", "text": "error_context", "count": number}]
                        }
                        
                        Be thorough but don't over-count. Context matters."""},
                        {"role": "user", "content": transcript}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3
                )
                
                analysis = json.loads(response.choices[0].message.content)
                
                # Calculate filler metrics
                total_fillers = sum(item["count"] for item in analysis["fillers"])
                fillers_per_minute = float(total_fillers / duration_minutes if duration_minutes > 0 else 0)
                
                # Calculate error metrics
                total_errors = sum(item["count"] for item in analysis["errors"])
                errors_per_minute = float(total_errors / duration_minutes if duration_minutes > 0 else 0)
                
            except Exception as e:
                logger.error(f"LLM analysis failed: {e}")
                # Fallback to simpler detection if LLM fails
                total_fillers = len(re.findall(r'\b(um|uh|like|you\s+know)\b', transcript.lower()))
                total_errors = len(re.findall(r'\b(\w+)\s+\1\b', transcript.lower()))
                fillers_per_minute = float(total_fillers / duration_minutes if duration_minutes > 0 else 0)
                errors_per_minute = float(total_errors / duration_minutes if duration_minutes > 0 else 0)
                analysis = {
                    "fillers": [{"word": "various", "count": total_fillers}],
                    "errors": [{"type": "repeated words", "count": total_errors}]
                }

            # Set thresholds
            max_errors = 1.0
            max_fillers = 3.0
            
            # Calculate fluency score
            fluency_score = 1 if (errors_per_minute <= max_errors and fillers_per_minute <= max_fillers) else 0
            
            # Get pitch variation coefficient from audio features
            pitch_variation_coeff = audio_features.get("pitch_variation_coeff", 0)
            
            return {
                "speed": {
                    "score": 1 if 120 <= words_per_minute <= 180 else 0,
                    "wpm": words_per_minute,
                    "total_words": words,
                    "duration_minutes": duration_minutes
                },
                "fluency": {
                    "score": fluency_score,
                    "errorsPerMin": errors_per_minute,
                    "fillersPerMin": fillers_per_minute,
                    "maxErrorsThreshold": max_errors,
                    "maxFillersThreshold": max_fillers,
                    "detectedErrors": [
                        {
                            "type": error["type"],
                            "context": error["text"] if "text" in error else "",
                            "count": error["count"]
                        } for error in analysis["errors"]
                    ],
                    "detectedFillers": [
                        {
                            "word": filler["word"],
                            "count": filler["count"]
                        } for filler in analysis["fillers"]
                    ]
                },
                "flow": {
                    "score": 1 if audio_features.get("pauses_per_minute", 0) <= 12 else 0,
                    "pausesPerMin": audio_features.get("pauses_per_minute", 0)
                },
                "intonation": {
                    "pitch": audio_features.get("pitch_mean", 0),
                    "pitchScore": 1 if 20 <= pitch_variation_coeff <= 40 else 0,
                    "pitchVariation": pitch_variation_coeff,  # Use pitch_variation_coeff here
                    "patternScore": 1 if audio_features.get("variations_per_minute", 0) >= 120 else 0,
                    "risingPatterns": audio_features.get("rising_patterns", 0),
                    "fallingPatterns": audio_features.get("falling_patterns", 0),
                    "variationsPerMin": audio_features.get("variations_per_minute", 0),
                    "mu": audio_features.get("pitch_mean", 0)
                },
                "energy": {
                    "score": 1 if 60 <= audio_features.get("mean_amplitude", 0) <= 75 else 0,
                    "meanAmplitude": audio_features.get("mean_amplitude", 0),
                    "amplitudeDeviation": audio_features.get("amplitude_deviation", 0),
                    "variationScore": 1 if 0.05 <= audio_features.get("amplitude_deviation", 0) <= 0.15 else 0
                }
            }

        except Exception as e:
            logger.error(f"Error in speech metrics evaluation: {e}")
            raise

    def generate_suggestions(self, category: str, citations: List[str]) -> List[str]:
        """Generate contextual suggestions based on category and citations"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """You are a teaching expert providing specific, actionable suggestions 
                    for improvement. Focus on the single most important, practical advice based on the teaching category 
                    and cited issues. Keep suggestions under 25 words."""},
                    {"role": "user", "content": f"""
                    Teaching Category: {category}
                    Issues identified in citations:
                    {json.dumps(citations, indent=2)}
                    
                    Please provide 2 or 3 at max specific, actionable suggestion for improvement.
                    Format as a JSON array with a single string."""}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("suggestions", [])
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return [f"Unable to generate specific suggestions: {str(e)}"]

class RecommendationGenerator:
    """Generates teaching recommendations using OpenAI API"""
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.retry_count = 3
        self.retry_delay = 1
        
    def generate_recommendations(self, 
                           metrics: Dict[str, Any], 
                           content_analysis: Dict[str, Any], 
                           progress_callback=None) -> Dict[str, Any]:
        """Generate recommendations with robust JSON handling"""
        try:
            if progress_callback:
                progress_callback(0.2, "Calculating hiring score...")
            
            # Calculate hiring score using our new manual function
            hiring_assessment = calculate_hiring_score(metrics)
            
            if progress_callback:
                progress_callback(0.5, "Generating recommendations...")
            
            # Use LLM only for generating the summary and other text-based recommendations
            prompt = self._create_recommendation_prompt(metrics, content_analysis, hiring_assessment)
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """You are a teaching expert providing actionable recommendations. 
                    Each improvement must be categorized as one of:
                    - COMMUNICATION: Related to speaking, pace, tone, clarity, delivery
                    - TEACHING: Related to explanation, examples, engagement, structure
                    - TECHNICAL: Related to code, implementation, technical concepts
                    
                    Always respond with a valid JSON object containing categorized improvements."""},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            if progress_callback:
                progress_callback(0.8, "Formatting recommendations...")
            
            result = json.loads(response.choices[0].message.content)
            
            # Replace the LLM-generated hiring recommendation with our manual one
            result["hiringRecommendation"] = {
                "score": hiring_assessment["score"],
                "reasons": hiring_assessment["reasons"],
                "strengths": hiring_assessment["strengths"],
                "concerns": hiring_assessment["concerns"]
            }
            
            if progress_callback:
                progress_callback(1.0, "Recommendations complete!")
            
            return result
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return {
                "summary": "Unable to generate recommendations due to an error",
                "geographyFit": "Unknown",
                "improvements": [
                    {
                        "category": "TECHNICAL",
                        "message": f"Error generating recommendations: {str(e)}"
                    }
                ],
                "rigor": "Undetermined",
                "profileMatches": [],
                "questionHandling": {
                    "confidence": "Unable to assess",
                    "accuracy": "Unable to assess",
                    "improvements": ["Error in recommendation generation"]
                },
                "hiringRecommendation": {
                    "score": 0,
                    "reasons": ["Error in recommendation generation"],
                    "strengths": [],
                    "concerns": ["System error prevented proper evaluation"]
                }
            }
    
    def _create_recommendation_prompt(self, metrics: Dict[str, Any], content_analysis: Dict[str, Any], hiring_assessment: Dict[str, Any]) -> str:
        """Create the recommendation prompt with scoring criteria"""
        # Get component scores from hiring assessment
        component_scores = hiring_assessment["component_scores"]
        
        return f"""Based on the following metrics and analysis, provide recommendations:

Component Scores:
1. Communication Score: {component_scores['communication']}/4
2. Teaching Score: {component_scores['teaching']}/4
3. QnA Score: {component_scores['qna']}/2
4. Total Score: {hiring_assessment['score']}/10

Assessment: {hiring_assessment['assessment']}
Description: {hiring_assessment['description']}

Analyze the teaching style and provide:
1. A simple and clear summary (3-5 short paragraphs)
2. Geography fit assessment
3. Specific improvements needed (each must be categorized as COMMUNICATION, TEACHING, or TECHNICAL)
4. Profile matching for different learner types (choose ONLY ONE best match)
5. Overall teaching rigor assessment
6. Question handling assessment (confidence, accuracy, and improvement areas)

Required JSON structure:
{{
    "summary": "Simple, clear summary using everyday language",
    "geographyFit": "String describing geographical market fit",
    "improvements": [
        {{
            "category": "COMMUNICATION/TEACHING/TECHNICAL",
            "message": "Specific improvement recommendation"
        }}
    ],
    "questionHandling": {{
        "confidence": "Assessment of confidence in answering questions",
        "accuracy": "Assessment of answer accuracy",
        "improvements": ["List of specific improvements"]
    }},
    "rigor": "Assessment of teaching rigor",
    "profileMatches": [
        {{
            "profile": "junior_technical",
            "match": false,
            "reason": "Simple explanation why this profile is not the best match"
        }},
        {{
            "profile": "senior_non_technical",
            "match": false,
            "reason": "Simple explanation why this profile is not the best match"
        }},
        {{
            "profile": "junior_expert",
            "match": false,
            "reason": "Simple explanation why this profile is not the best match"
        }},
        {{
            "profile": "senior_expert",
            "match": false,
            "reason": "Simple explanation why this profile is not the best match"
        }}
    ]
}}"""

class CostCalculator:
    """Calculates API and processing costs"""
    def __init__(self):
        self.GPT4_INPUT_COST = 0.15 / 1_000_000  # $0.15 per 1M tokens input
        self.GPT4_OUTPUT_COST = 0.60 / 1_000_000  # $0.60 per 1M tokens output
        self.WHISPER_COST = 0.006 / 60  # $0.006 per minute
        self.costs = {
            'transcription': 0.0,
            'content_analysis': 0.0,
            'recommendations': 0.0,
            'total': 0.0
        }

    def estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count based on words"""
        return len(text.split()) * 1.3  # Approximate tokens per word

    def add_transcription_cost(self, duration_seconds: float):
        """Calculate Whisper transcription cost"""
        cost = (duration_seconds / 60) * self.WHISPER_COST
        self.costs['transcription'] = cost
        self.costs['total'] += cost
        print(f"\nTranscription Cost: ${cost:.4f}")

    def add_gpt4_cost(self, input_text: str, output_text: str, operation: str):
        """Calculate GPT-4 API cost for a single operation"""
        input_tokens = self.estimate_tokens(input_text)
        output_tokens = self.estimate_tokens(output_text)
        
        input_cost = input_tokens * self.GPT4_INPUT_COST
        output_cost = output_tokens * self.GPT4_OUTPUT_COST
        total_cost = input_cost + output_cost
        
        self.costs[operation] = total_cost
        self.costs['total'] += total_cost
        
        print(f"\n{operation.replace('_', ' ').title()} Cost:")
        print(f"Input tokens: {input_tokens:.0f} (${input_cost:.4f})")
        print(f"Output tokens: {output_tokens:.0f} (${output_cost:.4f})")
        print(f"Operation total: ${total_cost:.4f}")

    def print_total_cost(self):
        """Print total cost breakdown"""
        print("\n=== Cost Breakdown ===")
        for key, cost in self.costs.items():
            if key != 'total':
                print(f"{key.replace('_', ' ').title()}: ${cost:.4f}")
        print(f"\nTotal Cost: ${self.costs['total']:.4f}")

class MentorEvaluator:
    """Main class for video evaluation"""
    def __init__(self, model_cache_dir: Optional[str] = None):
        # Fix potential API key issue
        self.api_key = st.secrets.get("OPENAI_API_KEY")  # Use get() method
        if not self.api_key:
            raise ValueError("OpenAI API key not found in secrets")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)  # Add this line
        
        # Add error handling for model cache directory
        try:
            if model_cache_dir:
                self.model_cache_dir = Path(model_cache_dir)
            else:
                self.model_cache_dir = Path.home() / ".cache" / "whisper"
            self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create model cache directory: {e}")
            
        # Initialize components with proper error handling
        try:
            self.feature_extractor = AudioFeatureExtractor()
            self.content_analyzer = ContentAnalyzer(self.api_key)
            self.recommendation_generator = RecommendationGenerator(self.api_key)
            self.cost_calculator = CostCalculator()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize components: {e}")

    def _get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result if available and not expired"""
        if key in self._cache:
            timestamp, value = self._cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return value
        return None

    def _set_cached_result(self, key: str, value: Any):
        """Cache result with timestamp"""
        self._cache[key] = (time.time(), value)

    def _extract_audio(self, video_path: str, output_path: str, progress_callback=None) -> str:
        """Extract audio with improved memory management"""
        try:
            if progress_callback:
                progress_callback(0.1, "Checking dependencies...")

            # Add optimized ffmpeg settings
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', video_path,
                '-ar', '16000',  # Set sample rate to 16kHz
                '-ac', '1',      # Convert to mono
                '-f', 'wav',     # Output format
                '-v', 'warning', # Reduce verbosity
                '-y',           # Overwrite output file
                '-c:a', 'pcm_s16le',  # Use simple audio codec
                '-movflags', 'faststart',  # Optimize for streaming
                '-threads', str(max(1, multiprocessing.cpu_count() - 1)),  # Use multiple threads
                output_path
            ]
            
            # Use subprocess with optimized buffer size
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                bufsize=10*1024*1024  # 10MB buffer
            )
            
            if result.returncode != 0:
                raise AudioProcessingError(f"FFmpeg Error: {result.stderr}")

            if not os.path.exists(output_path):
                raise AudioProcessingError("Audio extraction failed: output file not created")

            if progress_callback:
                progress_callback(1.0, "Audio extraction complete!")

            return output_path

        except Exception as e:
            logger.error(f"Error in audio extraction: {e}")
            raise AudioProcessingError(f"Audio extraction failed: {str(e)}")

    def _preprocess_audio(self, input_path: str, output_path: Optional[str] = None) -> str:
        """Preprocess audio for analysis"""
        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input audio file not found: {input_path}")

            # If no output path specified, use the input path
            if output_path is None:
                output_path = input_path

            # Load audio
            audio, sr = librosa.load(input_path, sr=16000)

            # Apply preprocessing steps
            # 1. Normalize audio
            audio = librosa.util.normalize(audio)

            # 2. Remove silence
            non_silent = librosa.effects.trim(audio, top_db=20)[0]

            # 3. Save processed audio
            sf.write(output_path, non_silent, sr)

            return output_path

        except Exception as e:
            logger.error(f"Error in audio preprocessing: {e}")
            raise AudioProcessingError(f"Audio preprocessing failed: {str(e)}")

    def evaluate_video(self, video_path: str, transcript_file: Optional[str] = None) -> Dict[str, Any]:
        try:
            # Add memory monitoring
            monitor_memory_usage()
            
            start_time = time.time()
            
            # Add input validation
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Validate video file format
            valid_extensions = {'.mp4', '.avi', '.mov'}
            if not any(video_path.lower().endswith(ext) for ext in valid_extensions):
                raise ValueError("Unsupported video format. Use MP4, AVI, or MOV")

            # Create progress tracking containers with error handling
            try:
                status = st.empty()
                progress = st.progress(0)
                tracker = ProgressTracker(status, progress)
            except Exception as e:
                logger.error(f"Failed to create progress trackers: {e}")
                raise

            # Add cleanup for temporary files
            temp_files = []
            try:
                with temporary_file(suffix=".wav") as temp_audio, \
                     temporary_file(suffix=".wav") as processed_audio:
                    temp_files.extend([temp_audio, processed_audio])
                    
                    # Step 1: Extract audio from video
                    tracker.update(0.1, "Extracting audio from video")
                    self._extract_audio(video_path, temp_audio)
                    tracker.next_step()
                    
                    # Step 2: Preprocess audio
                    tracker.update(0.2, "Preprocessing audio")
                    self._preprocess_audio(temp_audio, processed_audio)
                    tracker.next_step()
                    
                    # Step 3: Extract features
                    tracker.update(0.4, "Extracting audio features")
                    audio_features = self.feature_extractor.extract_features(processed_audio)
                    tracker.next_step()
                    
                    # Step 4: Get transcript - Modified to handle 3-argument progress callback
                    tracker.update(0.6, "Processing transcript")
                    if transcript_file:
                        transcript = transcript_file.getvalue().decode('utf-8')
                    else:
                        # Update progress callback to handle 3 arguments
                        tracker.update(0.6, "Transcribing audio")
                        transcript = self._transcribe_audio(
                            processed_audio, 
                            lambda p, m, extra=None: tracker.update(0.6 + p * 0.2, m)
                        )
                    tracker.next_step()
                    
                    # Step 5: Analyze content
                    tracker.update(0.8, "Analyzing teaching content")
                    content_analysis = self.content_analyzer.analyze_content(transcript)
                    
                    # Step 6: Generate recommendations
                    tracker.update(0.9, "Generating recommendations")
                    recommendations = self.recommendation_generator.generate_recommendations(
                        audio_features,
                        content_analysis
                    )
                    tracker.next_step()

                    # Add speech metrics evaluation
                    speech_metrics = self._evaluate_speech_metrics(transcript, audio_features)
                    
                    # Clear progress indicators
                    status.empty()
                    progress.empty()
                    
                    end_time = time.time()  # End timing
                    duration = end_time - start_time
                    
                    # Clear GPU memory after major operations
                    clear_gpu_memory()
                    
                    # Monitor memory again
                    monitor_memory_usage()
                    
                    return {
                        "audio_features": audio_features,
                        "transcript": transcript,
                        "teaching": content_analysis,
                        "recommendations": recommendations,
                        "speech_metrics": speech_metrics,
                        "processing_time": duration  # Add processing time to results
                    }

            finally:
                # Clean up any remaining temporary files
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary file {temp_file}: {e}")

        except Exception as e:
            logger.error(f"Error in video evaluation: {e}")
            # Clear memory on error
            clear_gpu_memory()
            raise

    def _transcribe_audio(self, audio_path: str, progress_callback=None) -> str:
        """Transcribe audio using Whisper with direct approach and timing"""
        try:
            if progress_callback:
                progress_callback(0.1, "Loading transcription model...")

            # Generate cache key based on file content
            cache_key = f"transcript_{hashlib.md5(open(audio_path, 'rb').read()).hexdigest()}"
            
            # Check cache first
            if cache_key in st.session_state:
                logger.info("Using cached transcription") 
                if progress_callback:
                    progress_callback(1.0, "Retrieved from cache")
                return st.session_state[cache_key]

            # Add validation for audio file
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            if progress_callback:
                progress_callback(0.2, "Downloading and initializing Whisper model (this may take a few minutes)...")

            try:
                # Load model with timeout and progress updates
                start_time = time.time()
                model = None
                
                def load_model():
                    nonlocal model
                    try:
                        model = whisper.load_model("medium")
                    except Exception as e:
                        logger.error(f"Error loading model: {e}")
                        raise

                # Create and start model loading thread
                model_thread = threading.Thread(target=load_model)
                model_thread.start()

                # Wait for model to load with progress updates
                while model_thread.is_alive():
                    elapsed = time.time() - start_time
                    if progress_callback:
                        progress_callback(0.2, f"Loading model... ({int(elapsed)}s elapsed)")
                    time.sleep(1)
                    
                    # Add timeout after 5 minutes
                    if elapsed > 300:  # 5 minutes timeout
                        raise TimeoutError("Model initialization timed out after 5 minutes")

                model_thread.join()
                
                if model is None:
                    raise RuntimeError("Model failed to initialize")

                if progress_callback:
                    progress_callback(0.4, "Model loaded successfully, starting transcription...")

                # Transcribe with progress updates
                result = model.transcribe(audio_path)
                transcript = result["text"]

                # Calculate elapsed time
                end_time = time.time()
                elapsed_time = end_time - start_time
                logger.info(f"Transcription completed in {elapsed_time:.2f} seconds")

                if progress_callback:
                    progress_callback(0.9, f"Transcription completed in {elapsed_time:.2f} seconds")

                # Validate transcript
                if not transcript.strip():
                    raise ValueError("Transcription produced empty result")

                # Cache the result
                st.session_state[cache_key] = transcript

                if progress_callback:
                    progress_callback(1.0, "Transcription complete!")

                return transcript

            except TimeoutError as te:
                logger.error(f"Model initialization timeout: {te}")
                raise RuntimeError("Model initialization timed out. Please try again.")
            except Exception as e:
                logger.error(f"Error during transcription: {e}")
                raise RuntimeError(f"Transcription failed: {str(e)}")

        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            if progress_callback:
                progress_callback(1.0, "Error in transcription", str(e))
            raise

    def calculate_speech_metrics(self, transcript: str, audio_duration: float) -> Dict[str, float]:
        """Calculate words per minute and other speech metrics."""
        words = len(transcript.split())
        minutes = audio_duration / 60
        return {
            'words_per_minute': words / minutes if minutes > 0 else 0,
            'total_words': words,
            'duration_minutes': minutes
        }

    def _evaluate_speech_metrics(self, transcript: str, audio_features: Dict[str, float], 
                               progress_callback=None) -> Dict[str, Any]:
        try:
            if progress_callback:
                progress_callback(0.2, "Calculating speech metrics...")

            # Calculate words and duration
            words = len(transcript.split())
            duration_minutes = float(audio_features.get('duration', 0)) / 60
            
            # Calculate words per minute
            words_per_minute = float(words / duration_minutes if duration_minutes > 0 else 0)
            
            # Use LLM to detect fillers and errors
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": """Analyze the speech transcript for:
                        1. Filler words (um, uh, like, you know, etc.)
                        2. Speech errors (repeated words, incomplete sentences, grammatical mistakes)
                        
                        Return a JSON with:
                        {
                            "fillers": [{"word": "filler_word", "count": number}],
                            "errors": [{"type": "error_type", "text": "error_context", "count": number}]
                        }
                        
                        Be thorough but don't over-count. Context matters."""},
                        {"role": "user", "content": transcript}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3
                )
                
                analysis = json.loads(response.choices[0].message.content)
                
                # Calculate filler metrics
                total_fillers = sum(item["count"] for item in analysis["fillers"])
                fillers_per_minute = float(total_fillers / duration_minutes if duration_minutes > 0 else 0)
                
                # Calculate error metrics
                total_errors = sum(item["count"] for item in analysis["errors"])
                errors_per_minute = float(total_errors / duration_minutes if duration_minutes > 0 else 0)
                
            except Exception as e:
                logger.error(f"LLM analysis failed: {e}")
                # Fallback to simpler detection if LLM fails
                total_fillers = len(re.findall(r'\b(um|uh|like|you\s+know)\b', transcript.lower()))
                total_errors = len(re.findall(r'\b(\w+)\s+\1\b', transcript.lower()))
                fillers_per_minute = float(total_fillers / duration_minutes if duration_minutes > 0 else 0)
                errors_per_minute = float(total_errors / duration_minutes if duration_minutes > 0 else 0)
                analysis = {
                    "fillers": [{"word": "various", "count": total_fillers}],
                    "errors": [{"type": "repeated words", "count": total_errors}]
                }

            # Set thresholds
            max_errors = 1.0
            max_fillers = 3.0
            
            # Calculate fluency score
            fluency_score = 1 if (errors_per_minute <= max_errors and fillers_per_minute <= max_fillers) else 0
            
            # Get pitch variation coefficient from audio features
            pitch_variation_coeff = audio_features.get("pitch_variation_coeff", 0)
            
            return {
                "speed": {
                    "score": 1 if 120 <= words_per_minute <= 180 else 0,
                    "wpm": words_per_minute,
                    "total_words": words,
                    "duration_minutes": duration_minutes
                },
                "fluency": {
                    "score": fluency_score,
                    "errorsPerMin": errors_per_minute,
                    "fillersPerMin": fillers_per_minute,
                    "maxErrorsThreshold": max_errors,
                    "maxFillersThreshold": max_fillers,
                    "detectedErrors": [
                        {
                            "type": error["type"],
                            "context": error["text"] if "text" in error else "",
                            "count": error["count"]
                        } for error in analysis["errors"]
                    ],
                    "detectedFillers": [
                        {
                            "word": filler["word"],
                            "count": filler["count"]
                        } for filler in analysis["fillers"]
                    ]
                },
                "flow": {
                    "score": 1 if audio_features.get("pauses_per_minute", 0) <= 12 else 0,
                    "pausesPerMin": audio_features.get("pauses_per_minute", 0)
                },
                "intonation": {
                    "pitch": audio_features.get("pitch_mean", 0),
                    "pitchScore": 1 if 20 <= pitch_variation_coeff <= 40 else 0,
                    "pitchVariation": pitch_variation_coeff,  # Use pitch_variation_coeff here
                    "patternScore": 1 if audio_features.get("variations_per_minute", 0) >= 120 else 0,
                    "risingPatterns": audio_features.get("rising_patterns", 0),
                    "fallingPatterns": audio_features.get("falling_patterns", 0),
                    "variationsPerMin": audio_features.get("variations_per_minute", 0),
                    "mu": audio_features.get("pitch_mean", 0)
                },
                "energy": {
                    "score": 1 if 60 <= audio_features.get("mean_amplitude", 0) <= 75 else 0,
                    "meanAmplitude": audio_features.get("mean_amplitude", 0),
                    "amplitudeDeviation": audio_features.get("amplitude_deviation", 0),
                    "variationScore": 1 if 0.05 <= audio_features.get("amplitude_deviation", 0) <= 0.15 else 0
                }
            }

        except Exception as e:
            logger.error(f"Error in speech metrics evaluation: {e}")
            raise

    def process_in_batches(self, audio: np.ndarray, sr: int, batch_duration: int = 300) -> List[Dict[str, Any]]:
        """Process audio in batches to manage memory better
        
        Args:
            audio: Audio array
            sr: Sample rate
            batch_duration: Duration of each batch in seconds
        
        Returns:
            List of results for each batch
        """
        try:
            # Calculate samples per batch
            samples_per_batch = sr * batch_duration
            total_samples = len(audio)
            
            results = []
            
            for start in range(0, total_samples, samples_per_batch):
                # Get batch
                end = min(start + samples_per_batch, total_samples)
                batch = audio[start:end]
                
                # Process batch
                batch_results = self.feature_extractor.extract_features(batch, sr)
                results.append(batch_results)
                
                # Clear memory after each batch
                clear_gpu_memory()
                gc.collect()
                
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise

def validate_video_file(file_path: str):
    """Validate video file before processing"""
    MAX_SIZE = 1024 * 1024 * 1024  # 1GB limit
    
    if not os.path.exists(file_path):
        raise ValueError("Video file does not exist")
    
    if os.path.getsize(file_path) > MAX_SIZE:
        raise ValueError(f"File size exceeds {MAX_SIZE/1024/1024}MB limit")
    
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    # Get the file extension and clean it
    file_ext = os.path.splitext(file_path)[1].lower().strip()
    
    # Log the actual extension for debugging
    logger.info(f"File extension detected: '{file_ext}'")
    
    if file_ext not in valid_extensions:
        raise ValueError(f"Unsupported video format: '{file_ext}'. Supported formats are: {', '.join(valid_extensions)}")
    
    # Additional validation using ffprobe
    try:
        # First try to get the format information
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1', file_path],
            capture_output=True,
            text=True
        )
        
        if probe.returncode != 0:
            # If that fails, try a basic probe
            probe = subprocess.run(
                ['ffprobe', '-v', 'quiet', file_path],
                capture_output=True,
                text=True
            )
            if probe.returncode != 0:
                raise ValueError("Invalid video file or corrupted video format")
        
        # Log the codec information for debugging
        codec = probe.stdout.strip()
        if codec:
            logger.info(f"Video codec detected: {codec}")
            
    except subprocess.SubprocessError as e:
        logger.error(f"FFprobe error: {str(e)}")
        raise ValueError("Unable to validate video file. Please ensure ffmpeg is installed and the video is not corrupted.")

def display_evaluation(evaluation: Dict[str, Any]):
    try:
        # Add a more prominent summary section at the top
        st.markdown("""
            <div class="summary-header">
                <h1>📊 Teaching Evaluation Results</h1>
                <p class="timestamp">Generated on: {}</p>
            </div>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)
        
        # Create a summary row with key metrics using a more prominent style
        col1, col2, col3, col4 = st.columns(4)
        
        # Communication Score with color coding
        with col1:
            speech_metrics = evaluation.get("speech_metrics", {})
            speed_data = speech_metrics.get("speed", {})
            words_per_minute = speed_data.get("wpm", 0)
            speed_score = "✅" if 120 <= words_per_minute <= 180 else "❌"
            st.markdown("""
                <div class="metric-box">
                    <h3>Communication</h3>
                    <div class="score">{}</div>
                    <div class="detail">WPM: {:.1f}</div>
                </div>
            """.format(speed_score, words_per_minute), unsafe_allow_html=True)
        
        # Teaching Score with color coding
        with col2:
            teaching_data = evaluation.get("teaching", {})
            concept_data = teaching_data.get("Concept Assessment", {})
            teaching_score = "✅" if all(d.get("Score", 0) == 1 for d in concept_data.values()) else "❌"
            st.markdown("""
                <div class="metric-box">
                    <h3>Teaching</h3>
                    <div class="score">{}</div>
                    <div class="detail">Concepts: {}/{} Pass</div>
                </div>
            """.format(teaching_score, sum(1 for d in concept_data.values() if d.get("Score", 0) == 1), len(concept_data)), unsafe_allow_html=True)
        
        # Overall Score with color coding
        with col3:
            overall_score = "✅" if speed_score == "✅" and teaching_score == "✅" else "❌"
            st.markdown("""
                <div class="metric-box">
                    <h3>Overall</h3>
                    <div class="score">{}</div>
                </div>
            """.format(overall_score), unsafe_allow_html=True)
        
        # Duration with formatting
        with col4:
            audio_features = evaluation.get("audio_features", {})
            duration = audio_features.get("duration", 0)
            st.markdown("""
                <div class="metric-box">
                    <h3>Duration</h3>
                    <div class="score">{:.1f} min</div>
                </div>
            """.format(duration), unsafe_allow_html=True)
        
        # Add custom CSS for better visibility
        st.markdown("""
        <style>
            .metric-box {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }
            .metric-box h3 {
                color: #1f77b4;
                margin-bottom: 10px;
            }
            .metric-box .score {
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
            }
            .metric-box .detail {
                color: #666;
                font-size: 14px;
            }
            .filler-card {
                background-color: #f8f9fa;
                border: 1px solid #e0e0e0;
                padding: 10px;
                margin: 5px;
                border-radius: 5px;
                text-align: center;
            }
            .filler-word {
                font-weight: bold;
                color: #1f77b4;
            }
            .filler-count {
                color: #666;
                font-size: 12px;
            }
            .error-context {
                background-color: #fff3f3;
                border-left: 4px solid #ff4444;
                padding: 10px;
                margin: 5px 0;
                border-radius: 4px;
            }
            .error-label {
                font-weight: bold;
                color: #ff4444;
            }
            .error-text {
                color: #666;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Add hover effects and better visual separation
        st.markdown("""
        <style>
            .metric-box {
                transition: transform 0.2s;
                cursor: pointer;
            }
            .metric-box:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .metric-box .score {
                font-size: 28px;
                margin: 15px 0;
            }
            .metric-box .detail {
                font-size: 16px;
                color: #666;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Improve the visibility of error contexts
        st.markdown("""
        <style>
            .error-context {
                background-color: #fff3f3;
                border-left: 4px solid #ff4444;
                padding: 15px;
                margin: 10px 0;
                border-radius: 4px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .error-label {
                font-weight: bold;
                color: #ff4444;
                margin-bottom: 5px;
            }
            .error-text {
                color: #666;
                font-size: 14px;
                line-height: 1.4;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Improve transcript readability
        st.markdown("""
        <style>
            .transcript-line {
                padding: 10px;
                margin: 5px 0;
                border-radius: 4px;
                background-color: #f8f9fa;
                transition: background-color 0.2s;
            }
            .transcript-line:hover {
                background-color: #e9ecef;
            }
            .timestamp {
                color: #666;
                font-weight: bold;
                margin-right: 10px;
                font-family: monospace;
            }
            .sentence {
                color: #333;
                line-height: 1.5;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Continue with existing tabs...
        tabs = st.tabs(["Communication", "Teaching", "Recommendations", "Transcript"])
        
        with tabs[0]:
            st.header("Communication Metrics")
            
            # Add a progress bar for overall communication score
            speech_metrics = evaluation.get("speech_metrics", {})
            speed_data = speech_metrics.get("speed", {})
            words_per_minute = speed_data.get("wpm", 0)
            
            # Calculate overall communication score
            speed_score = 1 if 120 <= words_per_minute <= 180 else 0
            fluency_data = speech_metrics.get("fluency", {})
            fillers_per_minute = float(fluency_data.get("fillersPerMin", 0))
            errors_per_minute = float(fluency_data.get("errorsPerMin", 0))
            fluency_score = 1 if fillers_per_minute <= 3 and errors_per_minute <= 1 else 0
            
            overall_score = (speed_score + fluency_score) / 2
            st.progress(overall_score, text="Overall Communication Score")
            
            # Continue with existing metrics...
            
            # Speed Metrics
            st.subheader("🏃 Speed")
            speed_data = speech_metrics.get("speed", {})
            words_per_minute = speed_data.get("wpm", 0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Score", "✅ Pass" if 120 <= words_per_minute <= 180 else "❌ Needs Improvement")
                st.metric("Words per Minute", f"{words_per_minute:.1f}")
            with col2:
                st.info("""
                **Acceptable Range:** 120-180 WPM
                - Optimal teaching pace: 130-150 WPM
                """)

            # Fluency Metrics
            st.subheader("🗣️ Fluency")
            speech_metrics = evaluation.get("speech_metrics", {})
            fluency_data = speech_metrics.get("fluency", {})
            
            fillers_per_minute = float(fluency_data.get("fillersPerMin", 0))
            errors_per_minute = float(fluency_data.get("errorsPerMin", 0))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Score", "✅ Pass" if fillers_per_minute <= 3 and errors_per_minute <= 1 else "❌ Needs Improvement")
                st.metric("Fillers per Minute", f"{fillers_per_minute:.1f}")
                st.metric("Errors per Minute", f"{errors_per_minute:.1f}")
            
            with col2:
                st.info("""
                **Acceptable Ranges:**
                - Fillers per Minute: <3
                - Errors per Minute: <1
                
                Analyzed using AI to detect context-appropriate usage.
                """)

            # Display filler words and errors in separate sections
            if "detectedFillers" in fluency_data:
                st.markdown("### 🗣️ Detected Filler Words")
                filler_data = fluency_data["detectedFillers"]
                
                if filler_data:
                    cols = st.columns(3)
                    for i, filler in enumerate(filler_data):
                        with cols[i % 3]:
                            st.markdown(
                                f"""
                                <div class="filler-card">
                                    <div class="filler-word">"{filler['word']}"</div>
                                    <div class="filler-count">Count: {filler['count']}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

            if "detectedErrors" in fluency_data:
                st.markdown("### ⚠️ Speech Errors")
                error_data = fluency_data["detectedErrors"]
                
                for error in error_data:
                    st.markdown(f"**{error['type']}** (Count: {error['count']})")
                    if 'context' in error and error['context']:
                        st.markdown("""
                            <div class="error-context">
                                <div class="error-label">Context:</div>
                                <div class="error-text">{}</div>
                            </div>
                        """.format(error['context']), unsafe_allow_html=True)

            # Continue with other metrics sections similarly...
            # Flow Metrics
            st.subheader("🌊 Flow")
            pauses_per_minute = float(audio_features.get("pauses_per_minute", 0))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Score", "✅ Pass" if pauses_per_minute <= 12 else "❌ Needs Improvement")
                st.metric("Pauses per Minute", f"{pauses_per_minute:.1f}")
            with col2:
                st.info("""
                **Acceptable Range:** 
                - Pauses per Minute: <12
                - Strategic pauses (8-12 PPM) aid comprehension
                """)

            # Intonation Metrics
            st.subheader("🎵 Intonation")
            pitch_mean = float(audio_features.get("pitch_mean", 0))
            pitch_std = float(audio_features.get("pitch_std", 0))
            pitch_variation_coeff = float(audio_features.get("pitch_variation_coeff", 0))
            monotone_score = float(audio_features.get("monotone_score", 0))
            direction_changes = float(audio_features.get("direction_changes_per_min", 0))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Monotone Score", f"{monotone_score:.2f}")
                st.metric("Pitch Variation", f"{pitch_variation_coeff:.1f}%")
                st.metric("Direction Changes/Min", f"{direction_changes:.1f}")
            with col2:
                # Add interpretation guide with stricter thresholds
                st.info("""
                **Monotone Analysis:**
                - Pitch Variation: 20-40% is optimal
                - Direction Changes: 300-600/min is optimal
                
                **Recommendations:**
                - Aim for pitch variation 20-40%
                - Target 300-600 direction changes/min
                - Use stress patterns for key points
                """)

                # Add visual indicator only for warning cases
                if monotone_score > 0.4 or pitch_variation_coeff < 20 or pitch_variation_coeff > 40 or direction_changes < 300 or direction_changes > 600:
                    st.warning("⚠️ Speech patterns need adjustment. Consider varying pitch and pace more naturally.")

            # Energy Metrics
            st.subheader("⚡ Energy")
            mean_amplitude = float(audio_features.get("mean_amplitude", 0))
            amplitude_deviation = float(audio_features.get("amplitude_deviation", 0))
            sigma_mu_ratio = float(amplitude_deviation) if mean_amplitude > 0 else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Amplitude", f"{mean_amplitude:.1f}")
                st.metric("Amplitude Deviation (σ)", f"{amplitude_deviation:.3f}")
            with col2:
                st.info("""
                **Acceptable Ranges:**
                - Mean Amplitude: 60-75
                - Amplitude Deviation: 0.05-0.15
                """)

        with tabs[1]:
            st.header("Teaching Analysis")
            
            # Add a radar chart for teaching metrics
            concept_data = teaching_data.get("Concept Assessment", {})
            categories = list(concept_data.keys())
            scores = [details.get("Score", 0) for details in concept_data.values()]
            
            # Create a radar chart using plotly
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=scores,
                theta=categories,
                fill='toself',
                name='Teaching Metrics'
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=False
            )
            st.plotly_chart(fig)
            
            # Continue with existing content...
            
            # Get teaching data from evaluation results
            teaching_data = evaluation.get("teaching", {})
            
            # Replace expanders with sections
            st.subheader("📚 Concept Assessment")
            concept_data = teaching_data.get("Concept Assessment", {})
            
            for category, details in concept_data.items():
                if category == "Question Handling":
                    continue
                
                score = details.get("Score", 0)
                citations = details.get("Citations", [])
                
                # Use columns for layout
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{category}**")
                with col2:
                    st.markdown("✅ Pass" if score == 1 else "❌ Needs Work")
                
                if citations:
                    st.markdown("##### 📝 Supporting Evidence")
                    for citation in citations:
                        st.markdown(f"- {citation}")

            # Code Assessment Section
            st.subheader("💻 Code Assessment")
            code_data = teaching_data.get("Code Assessment", {})
            
            for category, details in code_data.items():
                score = details.get("Score", 0)
                citations = details.get("Citations", [])
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{category}**")
                with col2:
                    st.markdown("✅ Pass" if score == 1 else "❌ Needs Work")
                
                if citations:
                    st.markdown("##### 📝 Supporting Evidence")
                    for citation in citations:
                        st.markdown(f"- {citation}")

        with tabs[2]:
            st.header("Recommendations")
            
            # Create a container for key recommendations
            with st.container():
                # Get recommendations from evaluation data first
                recommendations = evaluation.get("recommendations", {})
                if not recommendations:
                    st.warning("No recommendations data available.")
                    return
                
                # Get hiring recommendation data
                hiring_rec = recommendations.get("hiringRecommendation", {})
                score = hiring_rec.get("score", 0)
                reasons = hiring_rec.get("reasons", [])
                strengths = hiring_rec.get("strengths", [])
                concerns = hiring_rec.get("concerns", [])

                # Overall Score and Reasons - Moved to top
                st.markdown("""
                    <div style='background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;'>
                        <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                            <h4 style='color: #1f77b4; font-size: 24px; margin: 0;'>🎯 Overall Assessment Score</h4>
                            <div style='margin-left: 20px; padding: 8px 20px; background-color: {color}; color: white; border-radius: 15px; font-size: 20px;'>
                                {score}/10
                            </div>
                        </div>
                        <div style='margin-top: 15px;'>
                            <h5 style='color: #666; font-size: 16px; margin-bottom: 10px;'>Key Reasons:</h5>
                            <ul style='list-style-type: none; padding-left: 0;'>
                                {reasons_list}
                            </ul>
                        </div>
                    </div>
                """.format(
                    color='#2ecc71' if score >= 7 else '#f1c40f' if score >= 5 else '#e74c3c',
                    score=score,
                    reasons_list="\n".join([f"<li style='color: #333; margin-bottom: 8px; font-size: 16px;'>• {r}</li>" for r in reasons])
                ), unsafe_allow_html=True)

                # Overall Summary
                summary = recommendations.get("summary", "No summary available")
                st.markdown("""
                    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                        <h3 style='color: #1f77b4; margin-bottom: 15px;'>📝 Overall Summary</h3>
                        <p style='color: #333; line-height: 1.6;'>{}</p>
                    </div>
                """.format(summary), unsafe_allow_html=True)

                # Get recommendations from evaluation data
                recommendations = evaluation.get("recommendations", {})
                summary = recommendations.get("summary", "No summary available")
                
                # Get teaching data for detailed assessment
                teaching_data = evaluation.get("teaching", {})
                concept_data = teaching_data.get("Concept Assessment", {})
                speech_metrics = evaluation.get("speech_metrics", {})
                audio_features = evaluation.get("audio_features", {})
                
                # Content Accuracy Assessment
                content_accuracy = concept_data.get("Subject Matter Accuracy", {})
                content_score = content_accuracy.get("Score", 0)
                content_citations = content_accuracy.get("Citations", [])
                
                # Industry Examples Assessment
                examples = concept_data.get("Examples and Business Context", {})
                examples_score = examples.get("Score", 0)
                examples_citations = examples.get("Citations", [])
                
                # Pace Assessment
                speed_data = speech_metrics.get("speed", {})
                words_per_minute = speed_data.get("wpm", 0)
                pace_score = 1 if 120 <= words_per_minute <= 180 else 0
                
                # QnA Assessment
                qna_data = concept_data.get("Question Handling", {})
                qna_score = qna_data.get("Score", 0)
                qna_details = qna_data.get("Details", {})
                response_accuracy = qna_details.get("ResponseAccuracy", {}).get("Score", 0)
                response_completeness = qna_details.get("ResponseCompleteness", {}).get("Score", 0)
                
                # Accent Assessment
                accent_info = audio_features.get("accent_classification", {})
                accent = accent_info.get("accent", "Unknown")
                accent_confidence = accent_info.get("confidence", 0)
                
                # Create a 2x3 grid for key aspects using columns
                col1, col2, col3 = st.columns(3)
                
                # Content Accuracy
                with col1:
                    st.markdown("""
                        <div style='background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <h4 style='color: #1f77b4; font-size: 16px;'>📚 Content Accuracy</h4>
                            <div style='display: flex; align-items: center; margin-top: 10px;'>
                                <span style='font-size: 20px; margin-right: 10px;'>{icon}</span>
                                <span style='color: #666;'>{label}</span>
                            </div>
                        </div>
                    """.format(
                        icon="✅" if content_score == 1 else "❌",
                        label="Accurate" if content_score == 1 else "Needs Review"
                    ), unsafe_allow_html=True)
                
                # Industry Examples
                with col2:
                    st.markdown("""
                        <div style='background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <h4 style='color: #1f77b4; font-size: 16px;'>💼 Industry Examples</h4>
                            <div style='display: flex; align-items: center; margin-top: 10px;'>
                                <span style='font-size: 20px; margin-right: 10px;'>{icon}</span>
                                <span style='color: #666;'>{label}</span>
                            </div>
                        </div>
                    """.format(
                        icon="✅" if examples_score == 1 else "❌",
                        label="Well Used" if examples_score == 1 else "Could Improve"
                    ), unsafe_allow_html=True)
                
                # Teaching Pace
                with col3:
                    st.markdown("""
                        <div style='background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <h4 style='color: #1f77b4; font-size: 16px;'>⏱️ Teaching Pace</h4>
                            <div style='display: flex; align-items: center; margin-top: 10px;'>
                                <span style='font-size: 20px; margin-right: 10px;'>{icon}</span>
                                <span style='color: #666;'>{label} WPM</span>
                            </div>
                        </div>
                    """.format(
                        icon="✅" if pace_score == 1 else "❌",
                        label=f"{words_per_minute:.1f}"
                    ), unsafe_allow_html=True)
                
                # Second row of metrics
                col4, col5, col6 = st.columns(3)
                
                # QnA Accuracy
                with col4:
                    st.markdown("""
                        <div style='background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <h4 style='color: #1f77b4; font-size: 16px;'>❓ QnA Accuracy</h4>
                            <div style='display: flex; align-items: center; margin-top: 10px;'>
                                <span style='font-size: 20px; margin-right: 10px;'>{icon}</span>
                                <span style='color: #666;'>{label}</span>
                            </div>
                        </div>
                    """.format(
                        icon="✅" if response_accuracy == 1 and response_completeness == 1 else "❌",
                        label="Accurate & Complete" if response_accuracy == 1 and response_completeness == 1 else "Needs Improvement"
                    ), unsafe_allow_html=True)
                
                # Accent Clarity
                with col5:
                    st.markdown("""
                        <div style='background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <h4 style='color: #1f77b4; font-size: 16px;'>🗣️ Accent Clarity</h4>
                            <div style='display: flex; align-items: center; margin-top: 10px;'>
                                <span style='font-size: 20px; margin-right: 10px;'>{icon}</span>
                                <span style='color: #666;'>{label}</span>
                            </div>
                        </div>
                    """.format(
                        icon="✅" if accent_confidence > 0.7 else "⚠️",
                        label=f"{accent} ({accent_confidence*100:.1f}% confidence)"
                    ), unsafe_allow_html=True)

                # Engagement Score
                with col6:
                    engagement_score = concept_data.get("Engagement and Interaction", {}).get("Score", 0)
                    st.markdown("""
                        <div style='background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <h4 style='color: #1f77b4; font-size: 16px;'>👥 Engagement</h4>
                            <div style='display: flex; align-items: center; margin-top: 10px;'>
                                <span style='font-size: 20px; margin-right: 10px;'>{icon}</span>
                                <span style='color: #666;'>{label}</span>
                            </div>
                        </div>
                    """.format(
                        icon="✅" if engagement_score == 1 else "❌",
                        label="Good Engagement" if engagement_score == 1 else "Low Engagement"
                    ), unsafe_allow_html=True)

                # Add a new row for additional metrics
                col7, col8, col9 = st.columns(3)

                # Fluency Score
                with col7:
                    fluency_data = speech_metrics.get("fluency", {})
                    fillers_per_min = float(fluency_data.get("fillersPerMin", 0))
                    errors_per_min = float(fluency_data.get("errorsPerMin", 0))
                    fluency_score = 1 if fillers_per_min <= 3 and errors_per_min <= 1 else 0
                    st.markdown("""
                        <div style='background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <h4 style='color: #1f77b4; font-size: 16px;'>🗣️ Fluency</h4>
                            <div style='display: flex; align-items: center; margin-top: 10px;'>
                                <span style='font-size: 20px; margin-right: 10px;'>{icon}</span>
                                <span style='color: #666;'>{label}</span>
                            </div>
                            <div style='color: #666; font-size: 12px; margin-top: 5px;'>
                                Fillers: {fillers:.1f}/min<br>
                                Errors: {errors:.1f}/min
                            </div>
                        </div>
                    """.format(
                        icon="✅" if fluency_score == 1 else "❌",
                        label="Good Flow" if fluency_score == 1 else "Needs Work",
                        fillers=fillers_per_min,
                        errors=errors_per_min
                    ), unsafe_allow_html=True)

                # Energy Level
                with col8:
                    energy_data = speech_metrics.get("energy", {})
                    mean_amplitude = float(energy_data.get("meanAmplitude", 0))
                    energy_score = 1 if 60 <= mean_amplitude <= 75 else 0
                    st.markdown("""
                        <div style='background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <h4 style='color: #1f77b4; font-size: 16px;'>⚡ Energy Level</h4>
                            <div style='display: flex; align-items: center; margin-top: 10px;'>
                                <span style='font-size: 20px; margin-right: 10px;'>{icon}</span>
                                <span style='color: #666;'>{label}</span>
                            </div>
                            <div style='color: #666; font-size: 12px; margin-top: 5px;'>
                                Amplitude: {amplitude:.1f}
                            </div>
                        </div>
                    """.format(
                        icon="✅" if energy_score == 1 else "❌",
                        label="Good Energy" if energy_score == 1 else "Needs Adjustment",
                        amplitude=mean_amplitude
                    ), unsafe_allow_html=True)

                                # Intonation
                with col9:
                    intonation_data = speech_metrics.get("intonation", {})
                    pitch_variation = float(intonation_data.get("pitchVariation", 0))  # This is already pitch_variation_coeff
                    pitch_mean = float(intonation_data.get("pitch", 0))
                    variation_coeff = pitch_variation  # Already calculated as coefficient in audio features
                    intonation_score = 1 if 20 <= variation_coeff <= 40 else 0
                    st.markdown("""
                        <div style='background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <h4 style='color: #1f77b4; font-size: 16px;'>🎵 Voice Variety</h4>
                            <div style='display: flex; align-items: center; margin-top: 10px;'>
                                <span style='font-size: 20px; margin-right: 10px;'>{icon}</span>
                                <span style='color: #666;'>{label}</span>
                            </div>
                            <div style='color: #666; font-size: 12px; margin-top: 5px;'>
                                Variation: {variation:.1f}%
                            </div>
                        </div>
                    """.format(
                        icon="✅" if intonation_score == 1 else "❌",
                        label="Good Variety" if intonation_score == 1 else "Too Monotone",
                        variation=variation_coeff
                    ), unsafe_allow_html=True)

                # After the metrics rows and before Detailed Summary, add Geography and Profile sections
                st.markdown("""
                    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 20px;'>
                        <h3 style='color: #1f77b4; margin-bottom: 15px;'>🌍 Market Fit & Teaching Style</h3>
                    </div>
                """, unsafe_allow_html=True)

                # Create two columns for Geography and Profile
                geo_col, profile_col = st.columns(2)

                # Geography Fit
                with geo_col:
                    geography_fit = recommendations.get("geographyFit", "Not Available")
                    accent_info = audio_features.get("accent_classification", {})
                    accent = accent_info.get("accent", "Unknown")
                    
                    # Accent label mapping
                    accent_labels = {
                        "0": "American", "1": "British", "2": "Chinese", "3": "Japanese",
                        "4": "Indian", "5": "Korean", "6": "Russian", "7": "Spanish",
                        "8": "French", "9": "German", "10": "Italian", "11": "Dutch",
                        "12": "Australian", "13": "Arabic", "14": "African", "15": "Other"
                    }
                    
                    # Get proper accent label
                    accent_display = accent.title() if not accent.isdigit() else accent_labels.get(accent, "Unknown")
                    
                    # Create accent information HTML
                    accent_html = f"""
                        <div style='margin-top: 15px; padding-top: 10px; border-top: 1px solid #eee;'>
                            <h5 style='color: #1f77b4; font-size: 14px; margin-bottom: 8px;'>🗣️ Accent Analysis</h5>
                            <p style='color: #333; margin-bottom: 5px;'>
                                <strong>Primary Accent:</strong> {accent_display}
                            </p>
                        </div>
                    """
                    
                    st.markdown(f"""
                        <div style='background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <h4 style='color: #1f77b4; font-size: 16px;'>🌍 Geography Fit</h4>
                            <p style='color: #333; margin-top: 10px;'>{geography_fit}</p>
                            {accent_html}
                        </div>
                    """, unsafe_allow_html=True)

                # Teaching Rigor and Profile Match
                with profile_col:
                    rigor = recommendations.get("rigor", "Not Available")
                    st.markdown("""
                        <div style='background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <h4 style='color: #1f77b4; font-size: 16px;'>📊 Teaching Style</h4>
                            <p style='color: #333; margin-top: 10px;'>{rigor}</p>
                        </div>
                    """.format(rigor=rigor), unsafe_allow_html=True)

                # Learner Profile Matches
                st.markdown("""
                    <div style='background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 20px;'>
                        <h4 style='color: #1f77b4; margin-bottom: 15px;'>👥 Best Suited For</h4>
                    </div>
                """, unsafe_allow_html=True)

                # Create columns for profile matches
                profile_cols = st.columns(2)
                
                # Get profile matches
                profile_matches = recommendations.get("profileMatches", [])
                
                # Split profiles between columns
                for i, profile in enumerate(profile_matches):
                    with profile_cols[i % 2]:
                        profile_name = profile.get("profile", "").replace("_", " ").title()
                        is_match = profile.get("match", False)
                        reason = profile.get("reason", "No reason provided")
                        
                        st.markdown(f"""
                            <div style='background-color: {'#f0f9ff' if is_match else '#ffffff'}; 
                                      padding: 15px; 
                                      border-radius: 8px; 
                                      box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
                                      margin-bottom: 10px;
                                      border-left: 4px solid {'#1f77b4' if is_match else '#e0e0e0'}'>
                                <div style='display: flex; align-items: center; margin-bottom: 8px;'>
                                    <span style='font-size: 20px; margin-right: 10px;'>
                                        {'✅' if is_match else '⚪'}
                                    </span>
                                    <span style='color: #1f77b4; font-weight: bold;'>{profile_name}</span>
                                </div>
                                <p style='color: #666; margin: 0; font-size: 14px;'>{reason}</p>
                            </div>
                        """, unsafe_allow_html=True)

                # Add spacing before Detailed Summary
                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

                # Detailed Summary
                st.markdown("""
                    <div style='background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 20px;'>
                        <h4 style='color: #1f77b4; margin-bottom: 15px;'>Detailed Assessment</h4>
                        <p style='color: #333; line-height: 1.6;'>{summary}</p>
                    </div>
                """.format(summary=summary), unsafe_allow_html=True)
                
                # Add specific recommendations if available
                if "improvements" in recommendations:
                    st.markdown("""
                        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 20px;'>
                            <h3 style='color: #1f77b4; margin-bottom: 15px;'>🎯 Specific Recommendations</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    for improvement in recommendations["improvements"]:
                        if isinstance(improvement, dict):
                            category = improvement.get("category", "General")
                            message = improvement.get("message", "")
                            st.markdown(f"""
                                <div style='background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 10px;'>
                                    <span style='color: #1f77b4; font-weight: bold;'>{category}</span>
                                    <p style='color: #333; margin-top: 5px;'>{message}</p>
                                </div>
                            """, unsafe_allow_html=True)

                # Add spacing before Nervousness Analysis
                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

                # Nervousness Analysis Section
                st.markdown("""
                    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 20px;'>
                        <h3 style='color: #1f77b4; margin-bottom: 15px;'>🎯 Confidence Assessment</h3>
                    </div>
                """, unsafe_allow_html=True)

                # Create columns for nervousness metrics
                nerv_col1, nerv_col2 = st.columns(2)

                with nerv_col1:
                    # Calculate nervousness indicators
                    fluency_data = speech_metrics.get("fluency", {})
                    fillers_per_min = float(fluency_data.get("fillersPerMin", 0))
                    errors_per_min = float(fluency_data.get("errorsPerMin", 0))
                    
                    # Speech pattern analysis
                    intonation_data = speech_metrics.get("intonation", {})
                    pitch_variation = float(intonation_data.get("pitchVariation", 0))  # This is already pitch_variation_coeff from audio_features
                    pitch_mean = float(intonation_data.get("pitch", 0))
                    
                    # Confidence indicators
                    filler_confidence = "High" if fillers_per_min <= 2 else "Medium" if fillers_per_min <= 4 else "Low"
                    error_confidence = "High" if errors_per_min <= 0.5 else "Medium" if errors_per_min <= 1 else "Low"
                    pitch_confidence = "High" if 20 <= pitch_variation <= 40 else "Low"  # Use pitch_variation directly since it's already a coefficient

                    st.markdown("""
                        <div style='background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <h4 style='color: #1f77b4; font-size: 16px;'>Speech Confidence Indicators</h4>
                            <div style='margin-top: 15px;'>
                                <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
                                    <span style='color: #666;'>Filler Words:</span>
                                    <span style='color: {color1};'>{filler_conf}</span>
                                </div>
                                <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
                                    <span style='color: #666;'>Speech Errors:</span>
                                    <span style='color: {color2};'>{error_conf}</span>
                                </div>
                                <div style='display: flex; justify-content: space-between;'>
                                    <span style='color: #666;'>Voice Control:</span>
                                    <span style='color: {color3};'>{pitch_conf}</span>
                                </div>
                            </div>
                        </div>
                    """.format(
                        color1='#2ecc71' if filler_confidence == "High" else '#f1c40f' if filler_confidence == "Medium" else '#e74c3c',
                        color2='#2ecc71' if error_confidence == "High" else '#f1c40f' if error_confidence == "Medium" else '#e74c3c',
                        color3='#2ecc71' if pitch_confidence == "High" else '#e74c3c',
                        filler_conf=filler_confidence,
                        error_conf=error_confidence,
                        pitch_conf=pitch_confidence
                    ), unsafe_allow_html=True)

                with nerv_col2:
                    # Overall confidence assessment
                    confidence_score = sum([
                        1 if filler_confidence == "High" else 0.5 if filler_confidence == "Medium" else 0,
                        1 if error_confidence == "High" else 0.5 if error_confidence == "Medium" else 0,
                        1 if pitch_confidence == "High" else 0
                    ]) / 3 * 100

                    st.markdown("""
                        <div style='background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <h4 style='color: #1f77b4; font-size: 16px;'>Overall Confidence Score</h4>
                            <div style='text-align: center; margin-top: 15px;'>
                                <div style='font-size: 36px; color: {color}; font-weight: bold;'>{score}%</div>
                                <div style='color: #666; margin-top: 10px;'>{assessment}</div>
                            </div>
                        </div>
                    """.format(
                        color='#2ecc71' if confidence_score >= 80 else '#f1c40f' if confidence_score >= 60 else '#e74c3c',
                        score=round(confidence_score),
                        assessment="Very Confident" if confidence_score >= 80 else "Moderately Confident" if confidence_score >= 60 else "Shows Nervousness"
                    ), unsafe_allow_html=True)

                # Detailed Justification Section
                st.markdown("""
                    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 20px;'>
                        <h3 style='color: #1f77b4; margin-bottom: 15px;'>📋 Detailed Justification</h3>
                    </div>
                """, unsafe_allow_html=True)

                # Get hiring recommendation data
                hiring_rec = recommendations.get("hiringRecommendation", {})
                score = hiring_rec.get("score", 0)
                reasons = hiring_rec.get("reasons", [])
                strengths = hiring_rec.get("strengths", [])
                concerns = hiring_rec.get("concerns", [])

                # Create columns for strengths and concerns
                just_col1, just_col2 = st.columns(2)

                with just_col1:
                    st.markdown("""
                        <div style='background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <h4 style='color: #2ecc71; font-size: 16px;'>✨ Key Strengths</h4>
                            <ul style='list-style-type: none; padding-left: 0; margin-top: 15px;'>
                                {strengths_list}
                            </ul>
                        </div>
                    """.format(
                        strengths_list="\n".join([f"<li style='color: #333; margin-bottom: 8px;'>• {s}</li>" for s in strengths])
                    ), unsafe_allow_html=True)

                with just_col2:
                    st.markdown("""
                        <div style='background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <h4 style='color: #e74c3c; font-size: 16px;'>🎯 Areas for Growth</h4>
                            <ul style='list-style-type: none; padding-left: 0; margin-top: 15px;'>
                                {concerns_list}
                            </ul>
                        </div>
                    """.format(
                        concerns_list="\n".join([f"<li style='color: #333; margin-bottom: 8px;'>• {c}</li>" for c in concerns])
                    ), unsafe_allow_html=True)

        with tabs[3]:
            st.header("Transcript with Timestamps")
            
            # Add a search box for the transcript
            search_term = st.text_input("Search in transcript", "")
            
            transcript = evaluation.get("transcript", "")
            sentences = re.split(r'(?<=[.!?])\s+', transcript)
            
            # Create a container for the transcript
            with st.container():
                for i, sentence in enumerate(sentences):
                    words_before = len(' '.join(sentences[:i]).split())
                    timestamp = words_before / 150
                    minutes = int(timestamp)
                    seconds = int((timestamp - minutes) * 60)
                    
                    # Highlight search term if present
                    if search_term and search_term.lower() in sentence.lower():
                        sentence = sentence.replace(
                            search_term,
                            f"<mark>{search_term}</mark>",
                            flags=re.IGNORECASE
                        )
                    
                    st.markdown(f"""
                    <div class="transcript-line">
                        <span class="timestamp">[{minutes:02d}:{seconds:02d}]</span>
                        <span class="sentence">{sentence}</span>
                    </div>
                    """, unsafe_allow_html=True)

    except Exception as e:
        logger.error(f"Error displaying evaluation: {e}")
        st.error(f"Error displaying results: {str(e)}")
        st.error("Please check the evaluation data structure and try again.")

    # Add this at the beginning of the display_evaluation function
    st.markdown("""
    <style>
        .recommendation-card {
            background-color: #f8f9fa;
            border-left: 4px solid #1f77b4;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .transcript-line {
            padding: 8px;
            margin: 4px 0;
            border-radius: 4px;
            background-color: #f8f9fa;
        }
        
        .timestamp {
            color: #666;
            font-weight: bold;
            margin-right: 10px;
        }
        
        .sentence {
            color: #333;
        }
        
        mark {
            background-color: #ffd700;
            padding: 2px;
            border-radius: 2px;
        }
        
        .metric-box {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
    </style>
    """, unsafe_allow_html=True)

    # Add hover effects and better visual separation
    st.markdown("""
    <style>
        .metric-box {
            transition: transform 0.2s;
            cursor: pointer;
        }
        .metric-box:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .metric-box .score {
            font-size: 28px;
            margin: 15px 0;
        }
        .metric-box .detail {
            font-size: 16px;
            color: #666;
        }
    </style>
    """, unsafe_allow_html=True)

    # Improve the visibility of error contexts
    st.markdown("""
    <style>
        .error-context {
            background-color: #fff3f3;
            border-left: 4px solid #ff4444;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .error-label {
            font-weight: bold;
            color: #ff4444;
            margin-bottom: 5px;
        }
        .error-text {
            color: #666;
            font-size: 14px;
            line-height: 1.4;
        }
    </style>
    """, unsafe_allow_html=True)

    # Improve transcript readability
    st.markdown("""
    <style>
        .transcript-line {
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
            background-color: #f8f9fa;
            transition: background-color 0.2s;
        }
        .transcript-line:hover {
            background-color: #e9ecef;
        }
        .timestamp {
            color: #666;
            font-weight: bold;
            margin-right: 10px;
            font-family: monospace;
        }
        .sentence {
            color: #333;
            line-height: 1.5;
        }
    </style>
    """, unsafe_allow_html=True)

def generate_pdf_report(evaluation_data: Dict[str, Any]) -> bytes:
    """Generate a more visually appealing and comprehensive PDF report."""
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                leftMargin=72, rightMargin=72,
                                topMargin=72, bottomMargin=72)
        styles = getSampleStyleSheet()

        # Extract all necessary data first
        teaching_data = evaluation_data.get("teaching", {})
        speech_metrics = evaluation_data.get("speech_metrics", {})
        audio_features = evaluation_data.get("audio_features", {})
        recommendations = evaluation_data.get("recommendations", {})

        # Content Assessment data
        concept_data = teaching_data.get("Concept Assessment", {})
        content_accuracy = concept_data.get("Subject Matter Accuracy", {})
        content_score = content_accuracy.get("Score", 0)

        # Industry Examples data
        examples = concept_data.get("Examples and Business Context", {})
        examples_score = examples.get("Score", 0)

        # Pace Assessment data
        speed_data = speech_metrics.get("speed", {})
        words_per_minute = speed_data.get("wpm", 0)
        pace_score = 1 if 120 <= words_per_minute <= 180 else 0

        # QnA Assessment data
        qna_data = concept_data.get("Question Handling", {})
        qna_details = qna_data.get("Details", {})
        response_accuracy = qna_details.get("ResponseAccuracy", {}).get("Score", 0)
        response_completeness = qna_details.get("ResponseCompleteness", {}).get("Score", 0)

        # Accent Assessment data
        accent_info = audio_features.get("accent_classification", {})
        accent = accent_info.get("accent", "Unknown")
        accent_confidence = accent_info.get("confidence", 0)

        # Fluency data
        fluency_data = speech_metrics.get("fluency", {})
        fillers_per_min = float(fluency_data.get("fillersPerMin", 0))
        errors_per_min = float(fluency_data.get("errorsPerMin", 0))

        # Intonation data
        intonation_data = speech_metrics.get("intonation", {})
        pitch_variation = float(intonation_data.get("pitchVariation", 0))
        pitch_mean = float(intonation_data.get("pitch", 0))

        # Energy data
        energy_data = speech_metrics.get("energy", {})
        mean_amplitude = float(energy_data.get("meanAmplitude", 0))

        # Define styles
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['h1'],
            fontSize=22,
            alignment=TA_CENTER,
            spaceAfter=20,
            textColor=colors.HexColor('#2c3e50')
        )
        heading1_style = ParagraphStyle(
            'SectionHeading',
            parent=styles['h2'],
            fontSize=16,
            spaceAfter=10,
            spaceBefore=12,
            textColor=colors.HexColor('#1f77b4'),
            borderPadding=5,
        )
        heading2_style = ParagraphStyle(
            'SubHeading',
            parent=styles['h3'],
            fontSize=13,
            spaceAfter=8,
            spaceBefore=8,
            textColor=colors.HexColor('#34495e')
        )
        body_style = ParagraphStyle(
            'BodyText',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            alignment=TA_LEFT
        )
        citation_style = ParagraphStyle(
            'CitationText',
            parent=styles['Normal'],
            fontSize=9,
            leading=12,
            textColor=colors.dimgray,
            leftIndent=15
        )
        table_header_style = ParagraphStyle(
            'TableHeader',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.whitesmoke,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        table_body_style = ParagraphStyle(
            'TableBody',
            parent=styles['Normal'],
            fontSize=9,
            alignment=TA_LEFT
        )
        table_body_centered_style = ParagraphStyle(
            'TableBodyCentered',
            parent=table_body_style,
            alignment=TA_CENTER
        )
        score_pass_style = ParagraphStyle(
             'ScorePass', parent=table_body_centered_style, textColor=colors.darkgreen, fontName='Helvetica-Bold'
        )
        score_fail_style = ParagraphStyle(
             'ScoreFail', parent=table_body_centered_style, textColor=colors.red, fontName='Helvetica-Bold'
        )
        score_style = ParagraphStyle(
            'Score',
            parent=styles['h1'],
            fontSize=24,
            alignment=TA_CENTER,
            spaceAfter=10,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        )

        # --- Helper Functions for PDF Generation ---
        def create_metric_table(title: str, data: Dict[str, Any], metric_keys: list, value_format: str = "{:.2f}"):
            """Creates a styled table for simple key-value metrics."""
            if not data:
                return [Paragraph(f"{title}: Data not available", body_style)]

            table_data = [[Paragraph(title, heading2_style), None]] # Span title across columns
            table_data.append([Paragraph('Metric', table_header_style), Paragraph('Value', table_header_style)])

            for key in metric_keys:
                raw_value = data.get(key, 'N/A')
                value_str = 'N/A'
                if isinstance(raw_value, (int, float)):
                    try:
                        value_str = value_format.format(raw_value)
                    except (ValueError, TypeError):
                        value_str = str(raw_value) # Fallback
                elif raw_value is not None:
                    value_str = str(raw_value)

                metric_name = key.replace('_', ' ').replace('PerMin', '/Min').title()
                table_data.append([
                    Paragraph(metric_name, table_body_style),
                    Paragraph(value_str, table_body_centered_style)
                ])

            if len(table_data) <= 2: # Only title and header rows
                 return [Paragraph(f"{title}: No relevant data found", body_style)]

            table = Table(table_data, colWidths=[200, 100])
            style = TableStyle([
                ('SPAN', (0, 0), (1, 0)),  # Span title
                ('BACKGROUND', (0, 1), (1, 1), colors.HexColor('#4a69bd')),  # Header background
                ('TEXTCOLOR', (0, 1), (1, 1), colors.whitesmoke),
                ('ALIGN', (0, 1), (1, 1), 'CENTER'),
                ('FONTNAME', (0, 1), (1, 1), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 1), (1, 1), 8),
                ('GRID', (0, 1), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 2), (-1, -1), [colors.whitesmoke, colors.white])
            ])
            table.setStyle(style)
            return [table, Spacer(1, 15)]

        def create_analysis_table(title: str, data: Dict[str, Any]):
            """Creates a styled table for analysis categories with scores and citations."""
            if not data:
                 return [Paragraph(f"{title}: Data not available", body_style)]

            elements = [Paragraph(title, heading2_style)]
            for category, details in data.items():
                if not isinstance(details, dict): continue # Skip if data is not a dictionary

                score = details.get("Score", None)
                citations = details.get("Citations", [])

                score_text = "N/A"
                score_style = table_body_centered_style
                if score == 1:
                    score_text = "Pass"
                    score_style = score_pass_style
                elif score == 0:
                    score_text = "Needs Work"
                    score_style = score_fail_style

                # Table for category score
                cat_table_data = [
                    [Paragraph(category.replace('_', ' ').title(), table_header_style), Paragraph('Score', table_header_style)],
                    [Paragraph(category.replace('_', ' ').title(), table_body_style), Paragraph(score_text, score_style)]
                 ]
                cat_table = Table(cat_table_data, colWidths=[300, 100])
                cat_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a69bd')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white) # Body background
                ]))
                elements.append(cat_table)

                # Add citations if present
                if citations:
                    elements.append(Paragraph("Supporting Evidence:", citation_style))
                    for citation in citations:
                        elements.append(Paragraph(f"• {citation}", citation_style))

                # Handle nested 'Details' structure for Question Handling
                if "Details" in details and isinstance(details["Details"], dict):
                    elements.append(Paragraph("Detailed Assessment:", heading2_style))
                    nested_elements = create_analysis_table("", details["Details"]) # Recursive call for details
                    elements.extend(nested_elements) # Add nested elements

                elements.append(Spacer(1, 10))

            return elements

        def create_recommendation_section(title: str, content: Any):
            """Creates a section for recommendations or summary."""
            elements = [Paragraph(title, heading2_style)]
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                         # Handle new categorized format
                         category = item.get("category", "General")
                         message = item.get("message", str(item))
                         elements.append(Paragraph(f"• <b>[{category}]</b> {message}", body_style))
                    else:
                         # Handle old string format
                         elements.append(Paragraph(f"• {item}", body_style)) # Simple bullet point
            elif isinstance(content, str):
                elements.append(Paragraph(content, body_style))
            else:
                 elements.append(Paragraph("N/A", body_style))
            elements.append(Spacer(1, 15))
            return elements

        # --- Build PDF Story ---
        story = []
        story.append(Paragraph("Mentor Demo Review System", title_style))
        story.append(Spacer(1, 20))

        # --- Recommendations & Summary ---
        story.append(Paragraph("Recommendations & Summary", heading1_style))
        recommendations = evaluation_data.get("recommendations", {})

        # Hiring Recommendation
        hiring_rec = recommendations.get("hiringRecommendation", {})
        if hiring_rec:
            score = hiring_rec.get("score", 0)
            reasons = hiring_rec.get("reasons", [])
            strengths = hiring_rec.get("strengths", [])
            concerns = hiring_rec.get("concerns", [])
            
            # Display score
            score_color = colors.darkgreen if score >= 7 else colors.orange if score >= 5 else colors.red
            score_style = ParagraphStyle(
                'HiringScore',
                parent=score_style,
                textColor=score_color
            )
            story.append(Paragraph(f"Overall Assessment Score: {score}/10", score_style))
            story.append(Spacer(1, 10))
            
            # Display key reasons
            story.append(Paragraph("Key Reasons:", heading2_style))
            for reason in reasons:
                story.append(Paragraph(f"• {reason}", body_style))
            story.append(Spacer(1, 15))
            
            # Display strengths and concerns in two columns
            strengths_data = [[Paragraph("Key Strengths", heading2_style)]]
            for strength in strengths:
                strengths_data.append([Paragraph(f"• {strength}", body_style)])
            
            concerns_data = [[Paragraph("Areas for Growth", heading2_style)]]
            for concern in concerns:
                concerns_data.append([Paragraph(f"• {concern}", body_style)])
            
            # Create tables for strengths and concerns
            t_strengths = Table(strengths_data, colWidths=[250])
            t_concerns = Table(concerns_data, colWidths=[250])
            
            # Create a table to hold both columns
            combined_data = [[t_strengths, t_concerns]]
            t_combined = Table(combined_data, colWidths=[250, 250], spaceBefore=10, spaceAfter=10)
            story.append(t_combined)

        # Summary
        summary = recommendations.get("summary", "N/A")
        story.extend(create_recommendation_section("Overall Summary", summary))

        # Geography Fit and Teaching Style
        story.append(Paragraph("Market Fit & Teaching Style", heading1_style))
        
        # Create two-column layout for Geography and Teaching Style
        geography_fit = recommendations.get("geographyFit", "Not Available")
        rigor = recommendations.get("rigor", "Not Available")
        
        # Add accent information to geography fit
        accent_info = audio_features.get("accent_classification", {})
        accent = accent_info.get("accent", "Unknown")
        
        # Accent label mapping (same as above)
        accent_labels = {
            "0": "American", "1": "British", "2": "Chinese", "3": "Japanese",
            "4": "Indian", "5": "Korean", "6": "Russian", "7": "Spanish",
            "8": "French", "9": "German", "10": "Italian", "11": "Dutch",
            "12": "Australian", "13": "Arabic", "14": "African", "15": "Other"
        }
        
        # Get proper accent label
        accent_display = accent.title() if not accent.isdigit() else accent_labels.get(accent, "Unknown")
        
        # Create accent information text
        accent_text = f"Primary Accent: {accent_display}"
        
        # Combine geography fit with accent information
        geography_text = f"{geography_fit}\n\n{accent_text}"
        
        geo_teach_data = [
            [Paragraph("Geography Fit & Accent Analysis", heading2_style), Paragraph("Teaching Style", heading2_style)],
            [Paragraph(geography_text, body_style), Paragraph(rigor, body_style)]
        ]
        
        t_geo_teach = Table(geo_teach_data, colWidths=[250, 250], spaceBefore=10, spaceAfter=10)
        t_geo_teach.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(t_geo_teach)
        story.append(Spacer(1, 20))

        # --- Key Metrics Overview ---
        story.append(Paragraph("Key Teaching Metrics", heading1_style))
        
        # Create metrics table
        metrics_data = [
            ['Metric', 'Status', 'Details'],
            ['Content Accuracy', 
             '✓' if content_score == 1 else '✗',
             'Accurate' if content_score == 1 else 'Needs Review'],
            ['Industry Examples',
             '✓' if examples_score == 1 else '✗',
             'Well Used' if examples_score == 1 else 'Could Improve'],
            ['Teaching Pace',
             '✓' if pace_score == 1 else '✗',
             f'{words_per_minute:.1f} WPM'],
            ['QnA Accuracy',
             '✓' if response_accuracy == 1 and response_completeness == 1 else '✗',
             'Accurate & Complete' if response_accuracy == 1 and response_completeness == 1 else 'Needs Improvement'],
            ['Accent Clarity',
             '✓' if accent_confidence > 0.7 else '!',
             f'{accent} ({accent_confidence*100:.1f}% confidence)']
        ]
        
        t = Table(metrics_data, colWidths=[200, 50, 200])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 20))

        # --- Confidence Assessment ---
        story.append(Paragraph("Confidence Assessment", heading1_style))
        
        # Calculate confidence metrics
        fluency_data = speech_metrics.get("fluency", {})
        fillers_per_min = float(fluency_data.get("fillersPerMin", 0))
        errors_per_min = float(fluency_data.get("errorsPerMin", 0))
        
        intonation_data = speech_metrics.get("intonation", {})
        pitch_variation = float(intonation_data.get("pitchVariation", 0))  # This is already pitch_variation_coeff
        pitch_mean = float(intonation_data.get("pitch", 0))
        
        filler_confidence = "High" if fillers_per_min <= 2 else "Medium" if fillers_per_min <= 4 else "Low"
        error_confidence = "High" if errors_per_min <= 0.5 else "Medium" if errors_per_min <= 1 else "Low"
        pitch_confidence = "High" if 20 <= pitch_variation <= 40 else "Low"  # Use pitch_variation directly
        
        confidence_score = sum([
            1 if filler_confidence == "High" else 0.5 if filler_confidence == "Medium" else 0,
            1 if error_confidence == "High" else 0.5 if error_confidence == "Medium" else 0,
            1 if pitch_confidence == "High" else 0
        ]) / 3 * 100
        
        # Create confidence indicators table
        confidence_data = [
            ['Confidence Indicator', 'Level', 'Assessment'],
            ['Filler Words', filler_confidence, f'{fillers_per_min:.1f} per minute'],
            ['Speech Errors', error_confidence, f'{errors_per_min:.1f} per minute'],
            ['Voice Control', pitch_confidence, f'{pitch_variation:.1f}% variation']
        ]
        
        t = Table(confidence_data, colWidths=[150, 100, 200])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        
        # Overall Confidence Score
        story.append(Spacer(1, 15))
        confidence_text = f"Overall Confidence Score: {round(confidence_score)}%"
        confidence_assessment = "Very Confident" if confidence_score >= 80 else "Moderately Confident" if confidence_score >= 60 else "Shows Nervousness"
        story.append(Paragraph(confidence_text, heading2_style))
        story.append(Paragraph(confidence_assessment, body_style))
        story.append(Spacer(1, 20))

        # --- Recommendations & Summary ---
        story.append(Paragraph("Recommendations & Summary", heading1_style))
        recommendations = evaluation_data.get("recommendations", {})

        # Summary
        summary = recommendations.get("summary", "N/A")
        story.extend(create_recommendation_section("Overall Summary", summary))

        # Hiring Recommendation
        hiring_rec = recommendations.get("hiringRecommendation", {})
        if hiring_rec:
            score = hiring_rec.get("score", 0)
            reasons = hiring_rec.get("reasons", [])
            strengths = hiring_rec.get("strengths", [])
            concerns = hiring_rec.get("concerns", [])
            
            # Display score
            score_color = colors.darkgreen if score >= 7 else colors.orange if score >= 5 else colors.red
            score_style = ParagraphStyle(
                'HiringScore',
                parent=score_style,
                textColor=score_color
            )
            story.append(Paragraph(f"Hiring Score: {score}/10", score_style))
            story.append(Spacer(1, 10))
            
            # Display strengths and concerns in two columns
            strengths_data = [[Paragraph("Key Strengths", heading2_style)]]
            for strength in strengths:
                strengths_data.append([Paragraph(f"• {strength}", body_style)])
            
            concerns_data = [[Paragraph("Areas for Growth", heading2_style)]]
            for concern in concerns:
                concerns_data.append([Paragraph(f"• {concern}", body_style)])
            
            # Create tables for strengths and concerns
            t_strengths = Table(strengths_data, colWidths=[250])
            t_concerns = Table(concerns_data, colWidths=[250])
            
            # Create a table to hold both columns
            combined_data = [[t_strengths, t_concerns]]
            t_combined = Table(combined_data, colWidths=[250, 250], spaceBefore=10, spaceAfter=10)
            story.append(t_combined)
            
            # Display reasons
            if reasons:
                story.append(Paragraph("Key Reasons:", heading2_style))
                for reason in reasons:
                    story.append(Paragraph(f"• {reason}", body_style))
                story.append(Spacer(1, 10))

        # Geography Fit and Teaching Style
        story.append(Paragraph("Market Fit & Teaching Style", heading1_style))
        
        # Create two-column layout for Geography and Teaching Style
        geography_fit = recommendations.get("geographyFit", "Not Available")
        rigor = recommendations.get("rigor", "Not Available")
        
        geo_teach_data = [
            [Paragraph("Geography Fit", heading2_style), Paragraph("Teaching Style", heading2_style)],
            [Paragraph(geography_fit, body_style), Paragraph(rigor, body_style)]
        ]
        
        t_geo_teach = Table(geo_teach_data, colWidths=[250, 250], spaceBefore=10, spaceAfter=10)
        t_geo_teach.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(t_geo_teach)
        
        # Learner Profile Matches
        story.append(Paragraph("Learner Profile Matches", heading1_style))
        profile_matches = recommendations.get("profileMatches", [])
        
        for profile in profile_matches:
            profile_name = profile.get("profile", "").replace("_", " ").title()
            is_match = profile.get("match", False)
            reason = profile.get("reason", "No reason provided")
            
            match_text = f"{'✓' if is_match else '✗'} {profile_name}"
            story.append(Paragraph(match_text, heading2_style))
            story.append(Paragraph(reason, body_style))
            story.append(Spacer(1, 10))

        # Continue with existing sections...
        # --- Communication Metrics ---
        story.append(Paragraph("Communication Metrics", heading1_style))
        speech_metrics = evaluation_data.get("speech_metrics", {})

        # Calculate acceptance status for each metric
        speed_data = speech_metrics.get("speed", {})
        fluency_data = speech_metrics.get("fluency", {})
        intonation_data = speech_metrics.get("intonation", {})

        # Get key metrics
        words_per_minute = float(speed_data.get("wpm", 0))
        fillers_per_min = float(fluency_data.get("fillersPerMin", 0))
        errors_per_min = float(fluency_data.get("errorsPerMin", 0))
        pitch_variation = float(intonation_data.get("pitchVariation", 0))

        # Define acceptance criteria for each metric
        speed_accepted = 120 <= words_per_minute <= 180
        fillers_accepted = fillers_per_min <= 3
        errors_accepted = errors_per_min <= 1
        pitch_accepted = pitch_variation >= 20

        # Create a table for metric acceptance status
        acceptance_data = [
            ['Metric', 'Status', 'Value', 'Target Range'],
            ['Speaking Pace', 
             '✓' if speed_accepted else '✗',
             f'{words_per_minute:.1f} WPM',
             '120-180 WPM'],
            ['Filler Words',
             '✓' if fillers_accepted else '✗',
             f'{fillers_per_min:.1f} per min',
             '≤ 3 per min'],
            ['Speech Errors',
             '✓' if errors_accepted else '✗',
             f'{errors_per_min:.1f} per min',
             '≤ 1 per min'],
            ['Pitch Variation',
             '✓' if pitch_accepted else '✗',
             f'{pitch_variation:.1f}%',
             '≥ 20%']
        ]

        # Create and style the acceptance table
        t = Table(acceptance_data, colWidths=[150, 50, 100, 150])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 15))

        # Overall acceptance status
        overall_accepted = all([speed_accepted, fillers_accepted, errors_accepted, pitch_accepted])
        acceptance_status = "Accepted" if overall_accepted else "Not Accepted"
        acceptance_color = colors.darkgreen if overall_accepted else colors.red

        acceptance_style = ParagraphStyle(
            'AcceptanceStatus',
            parent=heading2_style,
            textColor=acceptance_color,
            fontSize=14,
            alignment=TA_CENTER
        )
        story.append(Paragraph(f"Overall Status: {acceptance_status}", acceptance_style))
        story.append(Spacer(1, 15))

        # Continue with detailed metrics...
        story.extend(create_metric_table(
            "Speed Details", speed_data, ['wpm', 'total_words', 'duration_minutes']
        ))
        
        story.extend(create_metric_table(
             "Fluency Details", fluency_data, ['errorsPerMin', 'fillersPerMin']
        ))
        
        # Add detected fillers/errors details
        if fluency_data:
            fillers = fluency_data.get("detectedFillers", [])
            errors = fluency_data.get("detectedErrors", [])
            if fillers:
                story.append(Paragraph("Detected Fillers:", heading2_style))
                for f in fillers:
                    story.append(Paragraph(f"• {f.get('word', 'N/A')}: {f.get('count', 'N/A')}", body_style))
                story.append(Spacer(1, 5))
            if errors:
                story.append(Paragraph("Detected Errors:", heading2_style))
                for e in errors:
                    story.append(Paragraph(
                        f"• {e.get('type', 'N/A')} (Count: {e.get('count', 'N/A')}): {e.get('context', '')}",
                        body_style
                    ))
                story.append(Spacer(1, 10))

        # Flow
        flow_data = speech_metrics.get("flow", {})
        story.extend(create_metric_table("Flow", flow_data, ['pausesPerMin']))

        # Intonation & Energy
        audio_features = evaluation_data.get("audio_features", {})
        intonation_data = speech_metrics.get("intonation", {})
        energy_data = speech_metrics.get("energy", {})

        # Intonation Table
        intonation_display_data = {
            "Monotone Score": audio_features.get("monotone_score"),
            "Pitch Mean (Hz)": audio_features.get("pitch_mean"),
            "Pitch Variation Coeff (%)": intonation_data.get("pitchVariation"),  # Use the consistent value
            "Direction Changes/Min": audio_features.get("direction_changes_per_min"),
        }
        story.extend(create_metric_table("Intonation", intonation_display_data, list(intonation_display_data.keys())))

        # Energy Table
        energy_display_data = {
            "Mean Amplitude": audio_features.get("mean_amplitude"),
            "Amplitude Deviation": audio_features.get("amplitude_deviation"),
        }
        story.extend(create_metric_table("Energy", energy_display_data, list(energy_display_data.keys())))

        # --- Teaching Analysis ---
        story.append(Paragraph("Teaching Analysis", heading1_style))
        teaching_data = evaluation_data.get("teaching", {})

        # Concept Assessment
        concept_assessment_data = teaching_data.get("Concept Assessment", {})
        story.extend(create_analysis_table("Concept Assessment", concept_assessment_data))

        # Code Assessment
        code_assessment_data = teaching_data.get("Code Assessment", {})
        story.extend(create_analysis_table("Code Assessment", code_assessment_data))

        story.append(Spacer(1, 15))

        # --- Build PDF ---
        doc.build(story)
        pdf_data = buffer.getvalue()
        buffer.close()
        logger.info("PDF report generated successfully.")
        return pdf_data

    except ImportError:
         logger.error("Reportlab not installed. Cannot generate PDF.")
         raise RuntimeError("PDF generation requires 'reportlab' library. Please install it (`pip install reportlab`).")
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}", exc_info=True) # Log traceback
        # Provide a more informative error message
        raise RuntimeError(f"Failed to generate PDF report: {str(e)}. Check logs for details.")

class GoogleDriveHandler:
    """Handles Google Drive operations for video processing"""
    SCOPES = ['https://www.googleapis.com/auth/drive']  # Updated scope to full drive access
    
    def __init__(self):
        self.creds = None
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Drive API using environment variables for Hugging Face Spaces"""
        try:
            # Check for credentials in environment variables
            credentials_json = os.getenv('GOOGLE_DRIVE_CREDENTIALS')
            if not credentials_json:
                raise ValueError("GOOGLE_DRIVE_CREDENTIALS environment variable not found")
            
            # Parse credentials from environment variable
            credentials_info = json.loads(credentials_json)
            
            # Create credentials from service account info with proper scopes
            self.creds = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=self.SCOPES
            )
            
            # Build the Drive API service
            self.service = build('drive', 'v3', credentials=self.creds)
            logger.info("Successfully authenticated with Google Drive")
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise RuntimeError(f"Failed to authenticate with Google Drive: {str(e)}")
    
    def list_videos_in_folder(self, folder_id: str) -> List[Dict[str, Any]]:
        """List all video files in a Google Drive folder with enhanced error handling"""
        try:
            # First verify folder exists and is accessible
            try:
                folder = self.service.files().get(
                    fileId=folder_id,
                    fields="id, name, mimeType"
                ).execute()
                logger.info(f"Successfully accessed folder: {folder.get('name')}")
            except Exception as e:
                logger.error(f"Error accessing folder: {str(e)}")
                raise ValueError(f"Could not access folder. Please check if the folder exists and you have permission to access it.")

            # List all files in the folder
            query = f"'{folder_id}' in parents"
            all_files = self.service.files().list(
                q=query,
                pageSize=100,
                fields="nextPageToken, files(id, name, mimeType)"
            ).execute()
            
            files = all_files.get('files', [])
            logger.info(f"Found {len(files)} total files in folder")
            
            # Filter for video files
            video_files = []
            for file in files:
                mime_type = file.get('mimeType', '')
                if mime_type.startswith('video/'):
                    video_files.append(file)
                else:
                    logger.info(f"Skipping non-video file: {file.get('name')} (MIME type: {mime_type})")
            
            logger.info(f"Found {len(video_files)} video files")
            
            if not video_files:
                # List all file types found for debugging
                mime_types = set(file.get('mimeType', '') for file in files)
                logger.info(f"Available file types in folder: {mime_types}")
                
                # Check if folder is empty
                if not files:
                    raise ValueError("The folder is empty")
                else:
                    raise ValueError(f"No video files found. The folder contains {len(files)} files of other types.")
            
            return video_files
            
        except Exception as e:
            logger.error(f"Error listing videos in folder: {str(e)}")
            raise
    
    def download_file(self, file_id: str, output_path: str):
        """Download a file from Google Drive with improved path handling"""
        try:
            # Ensure the directory exists
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Get file metadata to check MIME type
            file_metadata = self.service.files().get(
                fileId=file_id,
                fields='name,mimeType'
            ).execute()
            
            # Map MIME types to extensions
            mime_to_ext = {
                'video/mp4': '.mp4',
                'video/x-msvideo': '.avi',
                'video/quicktime': '.mov',
                'video/x-matroska': '.mkv',
                'video/webm': '.webm'
            }
            
            # Get the appropriate extension based on MIME type
            mime_type = file_metadata.get('mimeType', '')
            file_ext = mime_to_ext.get(mime_type, '')
            
            # If no extension found in MIME type mapping, try to get it from the filename
            if not file_ext:
                original_ext = os.path.splitext(file_metadata.get('name', ''))[1].lower()
                if original_ext in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}:
                    file_ext = original_ext
            
            # Sanitize the filename and add extension if needed
            filename = os.path.basename(output_path)
            sanitized_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            if not sanitized_filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                sanitized_filename += file_ext
            output_path = os.path.join(output_dir, sanitized_filename)
            
            logger.info(f"Downloading file to: {output_path}")
            
            request = self.service.files().get_media(
                fileId=file_id
            )
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            
            while not done:
                status, done = downloader.next_chunk()
                logger.info(f"Download {int(status.progress() * 100)}%")
            
            fh.seek(0)
            
            # Write the file with proper error handling
            try:
                with open(output_path, 'wb') as f:
                    f.write(fh.read())
                logger.info(f"Successfully downloaded file to: {output_path}")
            except IOError as e:
                logger.error(f"Failed to write file to {output_path}: {e}")
                raise
            
            # Verify the file exists and has content
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"File was not created at {output_path}")
            
            if os.path.getsize(output_path) == 0:
                raise ValueError(f"Downloaded file is empty: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            raise
    
    def upload_file(self, file_path: str, folder_id: str, mime_type: str = None) -> str:
        """Upload a file to Google Drive folder"""
        try:
            file_metadata = {
                'name': os.path.basename(file_path),
                'parents': [folder_id]
            }
            
            media = MediaFileUpload(
                file_path,
                mimetype=mime_type,
                resumable=True
            )
            
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            return file.get('id')
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            raise

def extract_folder_id_from_link(link: str) -> str:
    """Extract folder ID from Google Drive link"""
    try:
        # Handle different Google Drive link formats
        if 'folders' in link:
            # Format: https://drive.google.com/drive/folders/FOLDER_ID
            folder_id = link.split('folders/')[-1].split('?')[0]
        elif 'id=' in link:
            # Format: https://drive.google.com/open?id=FOLDER_ID
            folder_id = link.split('id=')[-1].split('&')[0]
        else:
            # Assume the link is just the ID
            folder_id = link.strip()
        
        # Validate folder ID format (typically 33 characters)
        if not re.match(r'^[a-zA-Z0-9_-]{33}$', folder_id):
            raise ValueError("Invalid folder ID format")
            
        return folder_id
    except Exception as e:
        raise ValueError(f"Could not extract folder ID from link: {str(e)}")

def extract_file_id_from_link(link: str) -> str:
    """Extract file ID from Google Drive file link or ID string"""
    try:
        # Handle different Google Drive file link formats
        if 'file/d/' in link:
            # Format: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
            file_id = link.split('file/d/')[-1].split('/')[0]
        elif 'id=' in link:
            # Format: https://drive.google.com/open?id=FILE_ID
            file_id = link.split('id=')[-1].split('&')[0]
        else:
            # Assume the link is just the ID
            file_id = link.strip()
        # Validate file ID format (typically 28-33 characters, but can vary)
        if not re.match(r'^[a-zA-Z0-9_-]{20,}$', file_id):
            raise ValueError("Invalid file ID format")
        return file_id
    except Exception as e:
        raise ValueError(f"Could not extract file ID from link: {str(e)}")

def handle_google_drive_analysis(input_folder_link: str, output_folder_link: str):
    """Handle analysis of videos from Google Drive folder using links"""
    try:
        # Extract folder IDs from links
        try:
            input_folder_id = extract_folder_id_from_link(input_folder_link)
            output_folder_id = extract_folder_id_from_link(output_folder_link)
            logger.info(f"Successfully extracted folder IDs - Input: {input_folder_id}, Output: {output_folder_id}")
        except ValueError as e:
            st.error(str(e))
            return
            
        # Check for required environment variables
        if not os.getenv('GOOGLE_DRIVE_CREDENTIALS'):
            st.error("Google Drive credentials not found in environment variables!")
            st.info("""
            Please set up your Google Drive credentials in Hugging Face Spaces:
            1. Go to your Space settings
            2. Add a new secret with key 'GOOGLE_DRIVE_CREDENTIALS'
            3. Paste your service account JSON credentials as the value
            """)
            return
            
        # Initialize Google Drive handler
        drive_handler = GoogleDriveHandler()
        
        # List videos in input folder with enhanced error handling
        try:
            videos = drive_handler.list_videos_in_folder(input_folder_id)
        except ValueError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Error accessing Google Drive: {str(e)}")
            return

        if not videos:
            st.warning("No videos found in the specified folder.")
            return
        
        # List existing reports in output folder with more detailed query
        query = f"'{output_folder_id}' in parents and mimeType='application/pdf' and trashed=false"
        existing_reports = drive_handler.service.files().list(
            q=query,
            pageSize=1000,  # Increased page size to ensure we get all reports
            fields="files(id, name, mimeType)"
        ).execute()
        
        # Create a set of normalized report names for case-insensitive comparison
        existing_report_names = set()
        for report in existing_reports.get('files', []):
            # Normalize the report name by:
            # 1. Converting to lowercase
            # 2. Removing _report.pdf suffix
            # 3. Removing date/time information and GMT references
            # 4. Removing special characters and extra spaces
            report_name = report.get('name', '')
            # Remove the _report.pdf suffix
            base_name = report_name.replace('_report.pdf', '')
            # Remove date/time information and GMT references
            base_name = re.sub(r' - \d{4}/\d{2}/\d{2} \d{2}:\d{2} GMT\+\d{2}:\d{2}', '', base_name)
            base_name = re.sub(r' - \d{4}_\d{2}_\d{2} \d{2}_\d{2} GMT\+\d{2}_\d{2}', '', base_name)
            # Remove any remaining special characters and normalize spaces
            normalized_name = re.sub(r'[^a-z0-9]', '_', base_name.lower())
            # Remove multiple consecutive underscores
            normalized_name = re.sub(r'_+', '_', normalized_name)
            # Remove leading/trailing underscores
            normalized_name = normalized_name.strip('_')
            existing_report_names.add(normalized_name)
            logger.info(f"Found existing report (normalized): {normalized_name}")
        
        # Filter videos that don't have reports yet
        videos_to_process = []
        skipped_videos = []
        
        for video in videos:
            video_name = video.get('name', '')
            # Normalize the video name using the same rules
            base_name = os.path.splitext(video_name)[0]
            # Remove date/time information and GMT references
            base_name = re.sub(r' - \d{4}/\d{2}/\d{2} \d{2}:\d{2} GMT\+\d{2}:\d{2}', '', base_name)
            base_name = re.sub(r' - \d{4}_\d{2}_\d{2} \d{2}_\d{2} GMT\+\d{2}_\d{2}', '', base_name)
            # Remove any remaining special characters and normalize spaces
            normalized_name = re.sub(r'[^a-z0-9]', '_', base_name.lower())
            # Remove multiple consecutive underscores
            normalized_name = re.sub(r'_+', '_', normalized_name)
            # Remove leading/trailing underscores
            normalized_name = normalized_name.strip('_')
            
            if normalized_name in existing_report_names:
                logger.info(f"Skipping {video_name} - report already exists (normalized name: {normalized_name})")
                skipped_videos.append(video_name)
            else:
                logger.info(f"Will process {video_name} - no existing report found (normalized name: {normalized_name})")
                videos_to_process.append(video)
        
        # Display summary of videos to process and skip
        if skipped_videos:
            st.info(f"Found {len(skipped_videos)} videos with existing reports:")
            for video_name in skipped_videos:
                st.markdown(f"- {video_name}")
        
        if not videos_to_process:
            st.success("All videos already have reports! No new videos to process.")
            return
        
        st.info(f"Found {len(videos_to_process)} new videos to process")
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize evaluator
        evaluator = MentorEvaluator()
        
        # Process each new video
        for i, video in enumerate(videos_to_process):
            try:
                video_name = video.get('name', '')
                status_text.text(f"Processing {video_name}...")
                
                # Create temp directory for processing with proper cleanup
                temp_dir = tempfile.mkdtemp()
                try:
                    # Get file metadata to check MIME type
                    file_metadata = drive_handler.service.files().get(
                        fileId=video['id'],
                        fields='name,mimeType'
                    ).execute()
                    
                    # Map MIME types to extensions
                    mime_to_ext = {
                        'video/mp4': '.mp4',
                        'video/x-msvideo': '.avi',
                        'video/quicktime': '.mov',
                        'video/x-matroska': '.mkv',
                        'video/webm': '.webm'
                    }
                    
                    # Get the appropriate extension based on MIME type
                    mime_type = file_metadata.get('mimeType', '')
                    file_ext = mime_to_ext.get(mime_type, '.mp4')  # Default to .mp4 if unknown
                    
                    # Sanitize the video filename and ensure it has the correct extension
                    safe_video_name = re.sub(r'[<>:"/\\|?*]', '_', video_name)
                    if not safe_video_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                        safe_video_name += file_ext
                    
                    video_path = os.path.join(temp_dir, safe_video_name)
                    logger.info(f"Processing video at path: {video_path}")
                    
                    # Download video
                    downloaded_path = drive_handler.download_file(video['id'], video_path)
                    
                    # Verify the video file exists and has content
                    if not os.path.exists(downloaded_path):
                        raise FileNotFoundError(f"Video file not found at {downloaded_path}")
                    if os.path.getsize(downloaded_path) == 0:
                        raise ValueError(f"Downloaded video file is empty: {downloaded_path}")
                    
                    # Process video
                    results = evaluator.evaluate_video(downloaded_path)
                    
                    # Generate PDF report
                    pdf_data = generate_pdf_report(results)
                    
                    # Save PDF temporarily
                    pdf_name = f"{os.path.splitext(safe_video_name)[0]}_report.pdf"
                    pdf_path = os.path.join(temp_dir, pdf_name)
                    with open(pdf_path, 'wb') as f:
                        f.write(pdf_data)
                    
                    # Upload PDF to output folder
                    drive_handler.upload_file(pdf_path, output_folder_id, 'application/pdf')
                    logger.info(f"Successfully uploaded report for {video_name}")
                    
                    # Append metrics to Google Sheet
                    append_metrics_to_sheet(results, video_name)
                    
                finally:
                    # Clean up temporary directory
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception as e:
                        logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")
                    
                    # Update progress
                    progress = (i + 1) / len(videos_to_process)
                    progress_bar.progress(progress)
                    
            except Exception as e:
                logger.error(f"Error processing {video_name}: {e}")
                st.error(f"Error processing {video_name}: {str(e)}")
                continue
        
        status_text.text("Processing complete!")
        st.success(f"Successfully processed {len(videos_to_process)} new videos")
        
    except Exception as e:
        logger.error(f"Error in Google Drive analysis: {e}")
        st.error(f"Error: {str(e)}")

def handle_single_video_analysis(input_type: str):
    """Handle analysis of a single video file"""
    try:
        # Initialize evaluator
        evaluator = MentorEvaluator()
        
        # Create file uploader based on input type
        if input_type == "Video Only (Auto-transcription)":
            video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
            transcript_file = None
        else:
            video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
            transcript_file = st.file_uploader("Upload Transcript (Optional)", type=['txt'])
        
        if video_file is not None:
            # Create progress tracking
            status = st.empty()
            progress = st.progress(0)
            tracker = ProgressTracker(status, progress)
            
            # Create temporary file for video
            with temporary_file(suffix=os.path.splitext(video_file.name)[1]) as temp_video:
                # Save uploaded video to temporary file
                with open(temp_video, 'wb') as f:
                    f.write(video_file.getvalue())
                
                try:
                    # Process video
                    results = evaluator.evaluate_video(temp_video, transcript_file)
                    
                    # Store results in session state
                    st.session_state.evaluation_results = results
                    st.session_state.processing_complete = True
                    
                    # Append metrics to Google Sheet
                    append_metrics_to_sheet(results, video_file.name)
                    
                    # Display results
                    display_evaluation(results)
                    
                    # Generate and offer PDF download
                    pdf_data = generate_pdf_report(results)
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_data,
                        file_name=f"{os.path.splitext(video_file.name)[0]}_report.pdf",
                        mime="application/pdf"
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing video: {e}")
                    st.error(f"Error processing video: {str(e)}")
                    st.session_state.processing_complete = False
    
    except Exception as e:
        logger.error(f"Error in single video analysis: {e}")
        st.error(f"Error: {str(e)}")

def handle_multiple_videos_analysis(input_type: str):
    """Handle analysis of multiple video files"""
    try:
        # Initialize evaluator
        evaluator = MentorEvaluator()
        
        # Create file uploader for multiple videos
        video_files = st.file_uploader("Upload Videos", type=['mp4', 'avi', 'mov'], accept_multiple_files=True)
        
        if video_files:
            # Create progress tracking
            status = st.empty()
            progress = st.progress(0)
            tracker = ProgressTracker(status, progress)
            
            # Process each video
            for i, video_file in enumerate(video_files):
                try:
                    status.text(f"Processing {video_file.name}...")
                    
                    # Create temporary file for video
                    with temporary_file(suffix=os.path.splitext(video_file.name)[1]) as temp_video:
                        # Save uploaded video to temporary file
                        with open(temp_video, 'wb') as f:
                            f.write(video_file.getvalue())
                        
                        # Process video
                        results = evaluator.evaluate_video(temp_video)
                        
                        # Generate PDF report
                        pdf_data = generate_pdf_report(results)
                        
                        # Offer PDF download
                        st.download_button(
                            label=f"Download Report for {video_file.name}",
                            data=pdf_data,
                            file_name=f"{os.path.splitext(video_file.name)[0]}_report.pdf",
                            mime="application/pdf"
                        )
                        
                        # Display results in expandable section
                        with st.expander(f"Results for {video_file.name}"):
                            display_evaluation(results)
                    
                    # Update progress
                    progress.progress((i + 1) / len(video_files))
                    
                except Exception as e:
                    logger.error(f"Error processing {video_file.name}: {e}")
                    st.error(f"Error processing {video_file.name}: {str(e)}")
                    continue
            
            status.text("Processing complete!")
            st.success(f"Successfully processed {len(video_files)} videos")
    
    except Exception as e:
        logger.error(f"Error in multiple videos analysis: {e}")
        st.error(f"Error: {str(e)}")

def handle_google_drive_single_file(file_link: str, output_folder_link: str):
    """Handle analysis of a single video from Google Drive file link and output folder link"""
    try:
        # Extract file ID and output folder ID
        try:
            file_id = extract_file_id_from_link(file_link)
            output_folder_id = extract_folder_id_from_link(output_folder_link)
            logger.info(f"Successfully extracted file ID: {file_id}, Output folder ID: {output_folder_id}")
        except ValueError as e:
            st.error(str(e))
            return

        # Check for required environment variables
        if not os.getenv('GOOGLE_DRIVE_CREDENTIALS'):
            st.error("Google Drive credentials not found in environment variables!")
            st.info("""
            Please set up your Google Drive credentials in Hugging Face Spaces:
            1. Go to your Space settings
            2. Add a new secret with key 'GOOGLE_DRIVE_CREDENTIALS'
            3. Paste your service account JSON credentials as the value
            """)
            return

        # Initialize Google Drive handler
        drive_handler = GoogleDriveHandler()

        # Get file metadata
        try:
            file_metadata = drive_handler.service.files().get(
                fileId=file_id,
                fields='name,mimeType'
            ).execute()
            video_name = file_metadata.get('name', f'{file_id}.mp4')
            mime_type = file_metadata.get('mimeType', 'video/mp4')
        except Exception as e:
            st.error(f"Could not access file: {str(e)}")
            return

        # Create temp directory for processing
        temp_dir = tempfile.mkdtemp()
        try:
            # Download video
            safe_video_name = re.sub(r'[<>:"/\\|?*]', '_', video_name)
            video_path = os.path.join(temp_dir, safe_video_name)
            downloaded_path = drive_handler.download_file(file_id, video_path)

            # Verify the video file exists and has content
            if not os.path.exists(downloaded_path):
                raise FileNotFoundError(f"Video file not found at {downloaded_path}")
            if os.path.getsize(downloaded_path) == 0:
                raise ValueError(f"Downloaded video file is empty: {downloaded_path}")

            # Initialize evaluator
            evaluator = MentorEvaluator()
            # Process video
            results = evaluator.evaluate_video(downloaded_path)

            # Generate PDF report
            pdf_data = generate_pdf_report(results)

            # Save PDF temporarily
            pdf_name = f"{os.path.splitext(safe_video_name)[0]}_report.pdf"
            pdf_path = os.path.join(temp_dir, pdf_name)
            with open(pdf_path, 'wb') as f:
                f.write(pdf_data)

            # Upload PDF to output folder
            drive_handler.upload_file(pdf_path, output_folder_id, 'application/pdf')
            logger.info(f"Successfully uploaded report for {video_name}")

            # Append metrics to Google Sheet
            append_metrics_to_sheet(results, video_name)

            st.success(f"Successfully processed and uploaded report for {video_name}")
        except Exception as e:
            logger.error(f"Error processing {video_name}: {e}")
            st.error(f"Error processing {video_name}: {str(e)}")
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in handle_google_drive_single_file: {e}")
        st.error(f"Unexpected error: {str(e)}")

def calculate_metric_score(metric_name: str, value: float) -> float:
    thresholds = METRIC_THRESHOLDS[metric_name]
    if 'excellent' in thresholds:
        if value <= thresholds['excellent']:
            return 1.0
        elif value <= thresholds['good']:
            return 0.7
        return 0.0
    else:
        if thresholds['min'] <= value <= thresholds['max']:
            return 1.0
        return 0.0

def assess_confidence(metrics: Dict[str, float]) -> Dict[str, str]:
    return {
        'filler_confidence': 'High' if metrics['fillers_per_min'] <= 1 else 
                           'Medium' if metrics['fillers_per_min'] <= 3 else 'Low',
        'error_confidence': 'High' if metrics['errors_per_min'] <= 0.2 else 
                          'Medium' if metrics['errors_per_min'] <= 1 else 'Low',
        'pitch_confidence': 'High' if 20 <= metrics['pitch_variation'] <= 40 else 'Low'
    }

def get_metric_display(metric_name: str, value: float) -> Dict[str, Any]:
    score = calculate_metric_score(metric_name, value)
    return {
        'score': score,
        'icon': '✅' if score >= 0.7 else '⚠️' if score >= 0.3 else '❌',
        'label': 'Excellent' if score == 1.0 else 'Good' if score >= 0.7 else 
                'Needs Improvement' if score >= 0.3 else 'Poor'
    }

def validate_metrics(metrics: Dict[str, float]) -> Dict[str, bool]:
    return {
        'monotone': metrics['monotone_score'] <= METRIC_THRESHOLDS['monotone_score']['good'],
        'pitch': METRIC_THRESHOLDS['pitch_variation']['min'] <= 
                metrics['pitch_variation'] <= METRIC_THRESHOLDS['pitch_variation']['max'],
        'direction': METRIC_THRESHOLDS['direction_changes']['min'] <= 
                    metrics['direction_changes'] <= METRIC_THRESHOLDS['direction_changes']['max'],
        'fillers': metrics['fillers_per_min'] <= METRIC_THRESHOLDS['fillers_per_min']['good'],
        'errors': metrics['errors_per_min'] <= METRIC_THRESHOLDS['errors_per_min']['good']
    }

def calculate_teaching_score(metrics: Dict[str, Any]) -> float:
    # Base weights for each component
    weights = {
        'communication': 0.4,  # 40% weight
        'teaching': 0.4,      # 40% weight
        'qna': 0.2           # 20% weight
    }
    
    # Calculate component scores (0-1 scale)
    comm_score = calculate_communication_score(metrics)
    teaching_score = calculate_teaching_component_score(metrics)
    qna_score = calculate_qna_score(metrics)
    
    # Calculate weighted total
    total_score = (
        comm_score * weights['communication'] +
        teaching_score * weights['teaching'] +
        qna_score * weights['qna']
    ) * 10  # Convert to 0-10 scale
    
    return total_score

def get_teaching_assessment(score: float) -> Dict[str, Any]:
    """Get teaching assessment based on updated thresholds"""
    if score >= TEACHING_ASSESSMENT_THRESHOLDS['excellent']:
        return {
            'rating': 'Excellent',
            'color': '#2ecc71',
            'icon': '✅',
            'description': 'Outstanding performance across all metrics'
        }
    elif score >= TEACHING_ASSESSMENT_THRESHOLDS['good']:
        return {
            'rating': 'Good',
            'color': '#27ae60',
            'icon': '✅', 
            'description': 'Strong performance with minor areas for improvement'
        }
    elif score >= TEACHING_ASSESSMENT_THRESHOLDS['acceptable']:
        return {
            'rating': 'Acceptable',
            'color': '#f1c40f',
            'icon': '⚠️',
            'description': 'Solid performance with some areas for improvement'
        }
    elif score >= TEACHING_ASSESSMENT_THRESHOLDS['needs_improvement']:
        return {
            'rating': 'Needs Improvement',
            'color': '#e67e22',
            'icon': '⚠️',
            'description': 'Several areas need improvement but shows potential'
        }
    else:
        return {
            'rating': 'Poor',
            'color': '#e74c3c',
            'icon': '❌',
            'description': 'Major improvements needed across multiple areas'
        }

def standardize_metric(metric_name: str, value: float) -> float:
    """Standardize metric values to 0-1 scale"""
    thresholds = METRIC_THRESHOLDS[metric_name]
    
    if 'excellent' in thresholds:
        if value <= thresholds['excellent']:
            return 1.0
        elif value <= thresholds['good']:
            return 0.7
        return 0.0
    else:
        min_val, max_val = thresholds['min'], thresholds['max']
        if min_val <= value <= max_val:
            return 1.0
        # Linear interpolation for values outside range
        if value < min_val:
            return max(0, 1 - (min_val - value) / min_val)
        return max(0, 1 - (value - max_val) / max_val)

def calculate_communication_score(metrics: Dict[str, Any]) -> float:
    speech_metrics = metrics.get("speech_metrics", {})
    comm_metrics = speech_metrics.get('intonation', {})
    monotone_score = comm_metrics.get('monotone_score', speech_metrics.get('monotone_score', 0))
    pitch_variation = comm_metrics.get('pitchVariation', comm_metrics.get('pitch_variation_coeff', 0))
    direction_changes = comm_metrics.get('direction_changes_per_min', 0)
    fillers_per_min = speech_metrics.get('fluency', {}).get('fillersPerMin', 0)
    errors_per_min = speech_metrics.get('fluency', {}).get('errorsPerMin', 0)
    words_per_minute = speech_metrics.get('speed', {}).get('wpm', 0)

    # More granular scoring for each metric
    comm_score_components = [
        max(0, 1 - (monotone_score / 0.3)) if monotone_score < 0.3 else 0,  # Gradual decrease from 1 to 0
        min(1, pitch_variation / 20),  # Linear scale up to 20
        min(1, direction_changes / 300),  # Linear scale up to 300
        max(0, 1 - (fillers_per_min / 3)),  # Gradual decrease from 1 to 0
        max(0, 1 - (errors_per_min / 1)),  # Gradual decrease from 1 to 0
        max(0, 1 - abs(words_per_minute - 140) / 20)  # Peak at 140 WPM, gradual decrease
    ]
    
    # Weight the components based on importance
    weights = [0.2, 0.2, 0.2, 0.15, 0.15, 0.1]  # Sum to 1
    weighted_score = sum(score * weight for score, weight in zip(comm_score_components, weights))
    return weighted_score

def calculate_teaching_component_score(metrics: Dict[str, Any]) -> float:
    teaching_data = metrics.get("teaching", {})
    concept_data = teaching_data.get("Concept Assessment", {})
    
    # Get detailed scores instead of binary
    subject_matter = concept_data.get('Subject Matter Accuracy', {}).get('Score', 0)
    examples = concept_data.get('Examples and Business Context', {}).get('Score', 0)
    qna = concept_data.get('Question Handling', {}).get('Score', 0)
    engagement = concept_data.get('Engagement and Interaction', {}).get('Score', 0)
    
    # Weight the components
    weights = [0.3, 0.3, 0.2, 0.2]  # Sum to 1
    components = [subject_matter, examples, qna, engagement]
    
    weighted_score = sum(score * weight for score, weight in zip(components, weights))
    return weighted_score

def calculate_qna_score(metrics: Dict[str, Any]) -> float:
    teaching_data = metrics.get("teaching", {})
    concept_data = teaching_data.get("Concept Assessment", {})
    qna_data = concept_data.get('Question Handling', {})
    qna_details = qna_data.get('Details', {})
    
    # Get detailed scores instead of binary
    response_accuracy = qna_details.get('ResponseAccuracy', {}).get('Score', 0)
    response_completeness = qna_details.get('ResponseCompleteness', {}).get('Score', 0)
    confidence_level = qna_details.get('ConfidenceLevel', {}).get('Score', 0)
    
    # Weight the components
    weights = [0.4, 0.4, 0.2]  # Sum to 1
    components = [response_accuracy, response_completeness, confidence_level]
    
    weighted_score = sum(score * weight for score, weight in zip(components, weights))
    return weighted_score

def calculate_hiring_score(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate hiring score manually based on metrics with more balanced and fair scoring.
    Returns a dictionary with score and assessment details.
    """
    # Get all required metrics
    speech_metrics = metrics.get("speech_metrics", {})
    teaching_data = metrics.get("teaching", {})
    concept_data = teaching_data.get("Concept Assessment", {})
    audio_features = metrics.get("audio_features", {})
    
    # 1. Communication Metrics (40% weight) - Restored original weight
    # Use more nuanced scoring for communication metrics
    wpm = speech_metrics.get('speed', {}).get('wpm', 0)
    fillers_per_min = speech_metrics.get('fluency', {}).get('fillersPerMin', 0)
    errors_per_min = speech_metrics.get('fluency', {}).get('errorsPerMin', 0)
    pitch_variation = speech_metrics.get('intonation', {}).get('pitchVariation', 0)
    monotone_score = audio_features.get('monotone_score', 0)
    
    # Debug logging to see what values we're getting
    logger.info(f"Communication metrics - WPM: {wpm}, Fillers: {fillers_per_min}, Errors: {errors_per_min}, Pitch: {pitch_variation}, Monotone: {monotone_score}")
    
    # More graduated scoring instead of binary with expanded WPM range
    comm_scores = {
        'monotone': 1.0 if monotone_score < 0.1 else 0.8 if monotone_score < 0.3 else 0.6 if monotone_score < 0.5 else 0.3,
        'pitch_variation': 1.0 if pitch_variation >= 25 else 0.8 if pitch_variation >= 20 else 0.6 if pitch_variation >= 15 else 0.3,
        'fillers': 1.0 if fillers_per_min <= 1 else 0.9 if fillers_per_min <= 2 else 0.7 if fillers_per_min <= 3 else 0.5 if fillers_per_min <= 4 else 0.3,
        'errors': 1.0 if errors_per_min <= 0.2 else 0.9 if errors_per_min <= 0.5 else 0.8 if errors_per_min <= 1 else 0.6 if errors_per_min <= 1.5 else 0.3,
        'pace': 1.0 if 130 <= wpm <= 150 else 0.9 if 120 <= wpm <= 180 else 0.7 if 110 <= wpm <= 190 else 0.5 if 100 <= wpm <= 200 else 0.3
    }
    comm_score = sum(comm_scores.values()) / len(comm_scores) * 4  # Convert to 0-4 scale
    
    # Debug logging
    logger.info(f"Individual communication scores: {comm_scores}")
    logger.info(f"Total communication score: {comm_score}/4")
    
    # 2. Teaching Metrics (40% weight) - Restored original weight
    teaching_scores = {
        'content_accuracy': 1 if concept_data.get('Subject Matter Accuracy', {}).get('Score', 0) == 1 else 0,
        'examples': 1 if concept_data.get('Examples and Business Context', {}).get('Score', 0) == 1 else 0,
        'storytelling': 1 if concept_data.get('Cohesive Storytelling', {}).get('Score', 0) == 1 else 0,
        'engagement': 1 if concept_data.get('Engagement and Interaction', {}).get('Score', 0) == 1 else 0,
        'professional_tone': 1 if concept_data.get('Professional Tone', {}).get('Score', 0) == 1 else 0
    }
    teaching_score = sum(teaching_scores.values()) / len(teaching_scores) * 4  # Convert to 0-4 scale
    
    # Debug logging
    logger.info(f"Teaching scores: {teaching_scores}")
    logger.info(f"Total teaching score: {teaching_score}/4")
    
    # 3. QnA Metrics (20% weight) - Same weight but more forgiving
    qna_data = concept_data.get('Question Handling', {})
    qna_details = qna_data.get('Details', {})
    qna_scores = {
        'response_accuracy': 1 if qna_details.get('ResponseAccuracy', {}).get('Score', 0) == 1 else 0.5,  # Partial credit
        'response_completeness': 1 if qna_details.get('ResponseCompleteness', {}).get('Score', 0) == 1 else 0.5,  # Partial credit
        'confidence': 1 if qna_details.get('ConfidenceLevel', {}).get('Score', 0) == 1 else 0.7  # More forgiving
    }
    qna_score = sum(qna_scores.values()) / len(qna_scores) * 2  # Convert to 0-2 scale
    
    # Debug logging
    logger.info(f"QnA scores: {qna_scores}")
    logger.info(f"Total QnA score: {qna_score}/2")
    
    # Calculate total score (0-10 scale)
    total_score = comm_score + teaching_score + qna_score
    
    # Debug logging
    logger.info(f"Final score calculation: {comm_score:.2f} + {teaching_score:.2f} + {qna_score:.2f} = {total_score:.2f}/10")
    
    # More nuanced assessment bands
    if total_score >= 8.5:
        assessment = "Excellent"
        color = "#2ecc71"  # Green
        icon = "✅"
        description = "Outstanding performance across all metrics"
    elif total_score >= 7.0:
        assessment = "Good"
        color = "#27ae60"  # Darker green
        icon = "✅"
        description = "Strong performance with minor areas for improvement"
    elif total_score >= 5.5:
        assessment = "Acceptable"
        color = "#f1c40f"  # Yellow
        icon = "⚠️"
        description = "Solid performance with some areas for improvement"
    elif total_score >= 4.0:
        assessment = "Needs Improvement"
        color = "#e67e22"  # Orange
        icon = "⚠️"
        description = "Several areas need improvement but shows potential"
    else:
        assessment = "Poor"
        color = "#e74c3c"  # Red
        icon = "❌"
        description = "Major improvements needed across multiple areas"
    
    # Generate reasons based on component scores with more positive framing
    reasons = []
    if teaching_score >= 4.0:  # 80% of teaching metrics passed
        reasons.append("Strong teaching methodology and content delivery")
    if comm_score >= 2.4:  # 80% of communication metrics
        reasons.append("Effective communication skills")
    elif comm_score >= 1.8:  # 60% of communication metrics  
        reasons.append("Good communication with room for refinement")
    if qna_score >= 1.6:  # 80% of QnA metrics
        reasons.append("Excellent question handling capabilities")
    
    # If overall score is good but individual components have issues
    if total_score >= 6.0 and len(reasons) < 2:
        reasons.append("Demonstrates core teaching competencies")
    
    # Generate strengths and concerns with more balanced view
    strengths = []
    concerns = []
    
    # Teaching strengths (prioritize since it's most important)
    if teaching_scores['content_accuracy'] == 1:
        strengths.append("Demonstrates strong subject matter expertise")
    if teaching_scores['examples'] == 1:
        strengths.append("Effectively uses practical examples and business context")
    if teaching_scores['engagement'] == 1:
        strengths.append("Shows good student engagement and interaction")
    if teaching_scores['storytelling'] == 1:
        strengths.append("Maintains cohesive and structured delivery")
    if teaching_scores['professional_tone'] == 1:
        strengths.append("Maintains professional and appropriate tone")
    
    # Communication strengths/concerns with graduated assessment
    if comm_scores['fillers'] >= 0.7:
        strengths.append("Maintains good speech fluency")
    elif comm_scores['fillers'] < 0.5:
        concerns.append("Could reduce use of filler words")
        
    if comm_scores['errors'] >= 0.8:
        strengths.append("Clear and accurate speech delivery")
    elif comm_scores['errors'] < 0.6:
        concerns.append("Could improve speech accuracy")
        
    if comm_scores['pace'] >= 0.7:
        if comm_scores['pace'] == 1.0:
            strengths.append("Excellent speaking pace for learning")
        else:
            strengths.append("Generally good speaking pace")
    else:
        if wpm > 160:
            concerns.append("Speaking pace could be slower for better comprehension")
        else:
            concerns.append("Speaking pace needs adjustment")
    
    if comm_scores['pitch_variation'] >= 0.8:
        strengths.append("Good voice modulation and variety")
    elif comm_scores['pitch_variation'] < 0.6:
        concerns.append("Could improve voice modulation for engagement")
    
    # QnA strengths/concerns
    if qna_scores['response_accuracy'] == 1:
        strengths.append("Provides accurate and reliable responses to questions")
    if qna_scores['response_completeness'] == 1:
        strengths.append("Gives comprehensive answers to student questions")
    if qna_scores['confidence'] >= 0.8:
        strengths.append("Shows confidence in handling questions")
    
    # Ensure we have at least some positive feedback if teaching is strong
    if len(strengths) == 0 and teaching_score >= 3.0:
        strengths.append("Shows solid teaching fundamentals")
        
    # Ensure concerns are constructive, not just negative
    if len(concerns) == 0 and total_score < 8.0:
        concerns.append("Continue developing communication techniques for even better delivery")
    
    return {
        "score": round(total_score, 1),
        "assessment": assessment,
        "color": color,
        "icon": icon,
        "description": description,
        "reasons": reasons,
        "strengths": strengths,
        "concerns": concerns,
        "component_scores": {
            "communication": round(comm_score, 1),
            "teaching": round(teaching_score, 1),
            "qna": round(qna_score, 1)
        }
    }

def main():
    try:
        # Initialize session state
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        if 'evaluation_results' not in st.session_state:
            st.session_state.evaluation_results = None
        
        # Add custom CSS
        st.markdown("""
            <style>
                /* ... existing styles ... */
            </style>
            
            <div class="fade-in">
                <h1 class="title-shimmer">
                    🎓 Mentor Demo Review System
                </h1>
            </div>
        """, unsafe_allow_html=True)

        # Sidebar with instructions
        with st.sidebar:
            st.markdown("""
                <div class="slide-in">
                    <h2>Instructions</h2>
                    <ol>
                        <li>Choose input method</li>
                        <li>Provide necessary credentials</li>
                        <li>Wait for the analysis</li>
                        <li>Review the results</li>
                    </ol>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Supported formats:** MP4, AVI, MOV")
            st.markdown("**Maximum file size:** 1GB")

        # Input method selection
        st.markdown("### 📤 Select Input Method")
        input_method = st.radio(
            "Choose how you want to provide videos:",
            options=[
                "Local Upload",
                "Google Drive Folder"
            ]
        )

        if input_method == "Google Drive Folder":
            st.markdown("### 🔑 Google Drive Configuration")
            
            # Check for credentials file
            if not os.path.exists('credentials.json'):
                st.error("Google Drive credentials file (credentials.json) not found!")
                st.info("Please place your Google Drive API credentials file in the same directory as this script.")
                return
            
            # New: Choose between folder and single file
            drive_mode = st.radio(
                "Choose Google Drive processing mode:",
                options=["Process Folder", "Process Single File"]
            )
            
            if drive_mode == "Process Folder":
                # Input folder link
                input_folder_link = st.text_input(
                    "Input Folder Link",
                    help="The Google Drive link to the folder containing videos to analyze"
                )
                
                # Output folder link
                output_folder_link = st.text_input(
                    "Output Folder Link",
                    help="The Google Drive link to the folder where reports will be saved"
                )
                
                if st.button("Start Processing (Folder)"):
                    if not input_folder_link or not output_folder_link:
                        st.warning("Please provide both input and output folder links.")
                    else:
                        handle_google_drive_analysis(input_folder_link, output_folder_link)
            else:
                # Single file mode
                file_link = st.text_input(
                    "Google Drive File Link",
                    help="Paste the Google Drive link to the video file to analyze"
                )
                output_folder_link = st.text_input(
                    "Output Folder Link",
                    help="The Google Drive link to the folder where the report will be saved"
                )
                if st.button("Start Processing (Single File)"):
                    if not file_link or not output_folder_link:
                        st.warning("Please provide both the file link and output folder link.")
                    else:
                        handle_google_drive_single_file(file_link, output_folder_link)
        
        else:
            # Existing local upload code
            st.markdown("### 🎥 Select Analysis Mode")
            analysis_mode = st.radio(
                "Choose how you want to analyze videos:",
                options=[
                    "Single Video Analysis",
                    "Multiple Videos Analysis"
                ]
            )
            
            st.markdown("### 📤 Select Upload Method")
            input_type = st.radio(
                "Choose how you want to provide your teaching content:",
                options=[
                    "Video Only (Auto-transcription)",
                    "Video + Manual Transcript"
                ]
            )
            
            if analysis_mode == "Single Video Analysis":
                handle_single_video_analysis(input_type)
            else:
                handle_multiple_videos_analysis(input_type)

    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    # Set Streamlit configuration for Hugging Face Spaces
    st.set_page_config(
        page_title="Mentor Demo Review System",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://huggingface.co/spaces',
            'Report a bug': None,
            'About': '# Mentor Demo Review System\nThis is a teaching evaluation system.'
        }
    )
    
    # Set environment variables for Hugging Face Spaces
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    
    # Add error handling for Google Drive credentials
    if 'GOOGLE_DRIVE_CREDENTIALS' not in os.environ:
        st.warning("""
        ⚠️ Google Drive credentials not found!
        
        To use Google Drive features:
        1. Create a Google Cloud service account
        2. Enable Google Drive API
        3. Create and download service account key
        4. Add the key to Hugging Face Space secrets as 'GOOGLE_DRIVE_CREDENTIALS'
        
        For now, you can use local file upload instead.
        """)
    
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Application error: {e}", exc_info=True)