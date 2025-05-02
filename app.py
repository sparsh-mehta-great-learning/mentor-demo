import streamlit as st
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

# Filter out ScriptRunContext warnings
warnings.filterwarnings('ignore', '.*ScriptRunContext!.*')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        """Evaluate speech metrics with LLM-based filler and error detection"""
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
            
            return {
                "speed": {
                    "score": 1 if 120 <= words_per_minute <= 160 else 0,
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
                    "pitchScore": 1 if 20 <= (audio_features.get("pitch_std", 0) / audio_features.get("pitch_mean", 0) * 100 if audio_features.get("pitch_mean", 0) > 0 else 0) <= 40 else 0,
                    "pitchVariation": audio_features.get("pitch_std", 0),
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
        for attempt in range(self.retry_count):
            try:
                if progress_callback:
                    progress_callback(0.2, "Preparing recommendation analysis...")
                
                prompt = self._create_recommendation_prompt(metrics, content_analysis)
                
                if progress_callback:
                    progress_callback(0.5, "Generating recommendations...")
                
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
                
                result_text = response.choices[0].message.content.strip()
                
                try:
                    result = json.loads(result_text)
                    # Ensure improvements are properly formatted
                    if "improvements" in result:
                        formatted_improvements = []
                        for imp in result["improvements"]:
                            if isinstance(imp, str):
                                # Default categorization for legacy format
                                formatted_improvements.append({
                                    "category": "TECHNICAL",
                                    "message": imp
                                })
                            elif isinstance(imp, dict):
                                # Ensure proper structure for dict format
                                formatted_improvements.append({
                                    "category": imp.get("category", "TECHNICAL"),
                                    "message": imp.get("message", str(imp))
                                })
                        result["improvements"] = formatted_improvements
                except json.JSONDecodeError:
                    result = {
                        "geographyFit": "Unknown",
                        "improvements": [
                            {
                                "category": "TECHNICAL",
                                "message": "Unable to generate specific recommendations"
                            }
                        ],
                        "rigor": "Undetermined",
                        "profileMatches": []
                    }
                
                if progress_callback:
                    progress_callback(1.0, "Recommendations complete!")
                
                return result
                
            except Exception as e:
                logger.error(f"Recommendation generation attempt {attempt + 1} failed: {e}")
                if attempt == self.retry_count - 1:
                    return {
                        "geographyFit": "Unknown",
                        "improvements": [
                            {
                                "category": "TECHNICAL",
                                "message": f"Unable to generate specific recommendations: {str(e)}"
                            }
                        ],
                        "rigor": "Undetermined",
                        "profileMatches": []
                    }
                time.sleep(self.retry_delay * (2 ** attempt))
    
    def _create_recommendation_prompt(self, metrics: Dict[str, Any], content_analysis: Dict[str, Any]) -> str:
        """Create the recommendation prompt"""
        return f"""Based on the following metrics and analysis, provide recommendations:
Metrics: {json.dumps(metrics)}
Content Analysis: {json.dumps(content_analysis)}

Analyze the teaching style and provide:
1. A simple and clear summary (2-3 short paragraphs) that:
   - Uses everyday language anyone can understand
   - Avoids technical terms and jargon
   - Clearly states what went well and what needs work
   - Includes these signs of nervousness in plain language:
     * Using filler words (um, uh, like) more than 3 times per minute
     * Making speech mistakes more than once per minute
     * Taking too many pauses (more than 12 per minute)
     * Speaking in a flat voice (monotone score above 0.4)
     * Speaking with unusual voice patterns
     * Speaking too quietly or too loudly
2. Geography fit assessment
3. Specific improvements needed (each must be categorized as COMMUNICATION, TEACHING, or TECHNICAL)
4. Profile matching for different learner types (choose ONLY ONE best match)
5. Overall teaching rigor assessment
6. Question handling assessment (confidence, accuracy, and improvement areas)

Required JSON structure:
{{
    "summary": "Simple, clear summary using everyday language that anyone can understand",
    "geographyFit": "String describing geographical market fit",
    "improvements": [
        {{
            "category": "COMMUNICATION",
            "message": "Specific improvement recommendation"
        }},
        {{
            "category": "TEACHING",
            "message": "Specific improvement recommendation"
        }},
        {{
            "category": "TECHNICAL",
            "message": "Specific improvement recommendation"
        }}
    ],
    "questionHandling": {{
        "confidence": "Assessment of confidence in answering questions",
        "accuracy": "Assessment of answer accuracy",
        "improvements": ["List of specific improvements for question handling"]
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
}}

Consider:
- How well the teaching flows
- If examples help explain things
- If code explanations make sense
- If students stay interested
- How well questions are answered
- How clearly they speak
- How well they teach
- If they seem nervous while teaching"""

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
        """Extract audio from video with optimized settings"""
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
                # Add these optimizations:
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
        start_time = time.time()  # Start timing
        try:
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
            end_time = time.time()  # Capture time even on error
            logger.error(f"Error in video evaluation (took {end_time - start_time:.2f}s): {e}")
            raise RuntimeError(f"Analysis failed: {str(e)}")

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
        """Evaluate speech metrics with LLM-based filler and error detection"""
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
            
            return {
                "speed": {
                    "score": 1 if 120 <= words_per_minute <= 160 else 0,
                    "wpm": words_per_minute,
                    "total_words": words,
                    "durat  ion_minutes": duration_minutes
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
                    "pitchScore": 1 if 20 <= (audio_features.get("pitch_std", 0) / audio_features.get("pitch_mean", 0) * 100 if audio_features.get("pitch_mean", 0) > 0 else 0) <= 40 else 0,
                    "pitchVariation": audio_features.get("pitch_std", 0),
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

def validate_video_file(file_path: str):
    """Validate video file before processing"""
    MAX_SIZE = 1024 * 1024 * 1024  # 500MB limit
    
    if os.path.getsize(file_path) > MAX_SIZE:
        raise ValueError(f"File size exceeds {MAX_SIZE/1024/1024}MB limit")
    
    valid_extensions = {'.mp4', '.avi', '.mov'}
    
    if not os.path.exists(file_path):
        raise ValueError("Video file does not exist")
        
    if os.path.splitext(file_path)[1].lower() not in valid_extensions:
        raise ValueError("Unsupported video format")
        
    try:
        probe = subprocess.run(
            ['ffprobe', '-v', 'quiet', file_path],
            capture_output=True,
            text=True
        )
        if probe.returncode != 0:
            raise ValueError("Invalid video file")
    except subprocess.SubprocessError:
        raise ValueError("Unable to validate video file")

def display_evaluation(evaluation: Dict[str, Any]):
    try:
        tabs = st.tabs(["Communication", "Teaching", "Recommendations", "Transcript"])
        
        with tabs[0]:
            st.header("Communication Metrics")
            
            # Get audio features and ensure we have the required metrics
            audio_features = evaluation.get("audio_features", {})
            
            # Create sections without nesting expanders
            st.subheader("🏃 Speed")
            # Speed Metrics
            speech_metrics = evaluation.get("speech_metrics", {})
            speed_data = speech_metrics.get("speed", {})
            words_per_minute = speed_data.get("wpm", 0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Score", "✅ Pass" if 120 <= words_per_minute <= 160 else "❌ Needs Improvement")
                st.metric("Words per Minute", f"{words_per_minute:.1f}")
            with col2:
                st.info("""
                **Acceptable Range:** 120-160 WPM
                - Optimal teaching pace: 130-160 WPM
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
                # st.metric("σ/μ Ratio", f"{sigma_mu_ratio:.3f}")
            with col2:
                st.info("""
                **Acceptable Ranges:**
                - Mean Amplitude: 60-75
                - Amplitude Deviation: 0.05-0.15
                """)

                # Add explanation card
                st.markdown("""
                <div class="metric-explanation-card">
                    <h4>📊 Understanding Energy Metrics</h4>
                    <ul>
                        <li><strong>Mean Amplitude:</strong> Average volume level of speech. 60-75 range ensures clear audibility without being too loud.</li>
                        <li><strong>Amplitude Deviation:</strong> Measures volume variation. 0.05-0.15 indicates good dynamic range without excessive fluctuation.</li>
                        <li><strong>Impact:</strong> Proper energy levels maintain listener engagement and emphasize key points without causing listener fatigue.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        with tabs[1]:
            st.header("Teaching Analysis")
            
            teaching_data = evaluation.get("teaching", {})
            content_analyzer = ContentAnalyzer(st.secrets["OPENAI_API_KEY"])
            
            # Display Concept Assessment with enhanced citation context
            with st.expander("📚 Concept Assessment", expanded=True):
                concept_data = teaching_data.get("Concept Assessment", {})
                
                for category, details in concept_data.items():
                    score = details.get("Score", 0)
                    citations = details.get("Citations", [])
                    
                    st.markdown(f"""
                        <div class="teaching-card">
                            <div class="teaching-header">
                                <span class="category-name">{category}</span>
                                <span class="score-badge {'score-pass' if score == 1 else 'score-fail'}">
                                    {'✅ Pass' if score == 1 else '❌ Needs Work'}
                                </span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if citations:
                        st.markdown("##### 📝 Supporting Evidence")
                        for citation in citations:
                            # Extract timestamp if available
                            timestamp_match = re.match(r'\[(\d+:\d+)\]', citation)
                            timestamp = timestamp_match.group(1) if timestamp_match else "00:00"
                            
                            # Extract the actual quote
                            quote = re.sub(r'\[\d+:\d+\]\s*', '', citation).strip("'")
                            
                            # Find the quote in the transcript
                            transcript = evaluation.get("transcript", "")
                            if transcript and quote in transcript:
                                # Get surrounding context (approximately 100 characters before and after)
                                quote_index = transcript.find(quote)
                                start_index = max(0, quote_index - 200)
                                end_index = min(len(transcript), quote_index + len(quote) + 200)
                                
                                # Get the context
                                context = transcript[start_index:end_index]
                                
                                # Add ellipsis if we truncated the context
                                if start_index > 0:
                                    context = "..." + context
                                if end_index < len(transcript):
                                    context = context + "..."
                                
                                # Highlight the quote within the context
                                highlighted_context = context.replace(quote, f'<span class="highlight">{quote}</span>')
                                
                                st.markdown(f"""
                                    <div class="citation-box">
                                        <div class="citation-timestamp">🕒 Timestamp: {timestamp}</div>
                                        <div class="citation-context">
                                            {highlighted_context}
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                # Fallback to displaying just the citation
                                st.markdown(f"""
                                    <div class="citation-box">
                                        <div class="citation-timestamp">🕒 Timestamp: {timestamp}</div>
                                        <div class="citation-text">"{quote}"</div>
                                    </div>
                                """, unsafe_allow_html=True)

            # Update the CSS for better citation display
            st.markdown("""
                <style>
                .citation-box {
                    background: #f8f9fa;
                    border-left: 4px solid #1f77b4;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 4px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    transition: transform 0.2s ease;
                }
                
                .citation-box:hover {
                    transform: translateX(5px);
                }
                
                .citation-timestamp {
                    color: #6c757d;
                    font-size: 0.9em;
                    font-weight: bold;
                    margin-bottom: 8px;
                }
                
                .citation-context {
                    color: #212529;
                    line-height: 1.6;
                    padding: 12px;
                    background: white;
                    border-radius: 4px;
                    border: 1px solid #e9ecef;
                    font-size: 1.1em;
                    margin-top: 8px;
                }
                
                .highlight {
                    background-color: #fff3cd;
                    padding: 2px 4px;
                    border-radius: 2px;
                    font-weight: bold;
                    color: #856404;
                }
                
                .citation-text {
                    font-style: italic;
                    color: #495057;
                    padding: 10px;
                    background: white;
                    border-radius: 4px;
                    border: 1px solid #e9ecef;
                }
                
                .teaching-card {
                    background: white;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 15px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                
                .teaching-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 10px;
                }
                
                .category-name {
                    font-size: 1.2em;
                    font-weight: bold;
                    color: #2c3e50;
                }
                
                .score-badge {
                    padding: 5px 10px;
                    border-radius: 15px;
                    font-weight: bold;
                }
                
                .score-pass {
                    background-color: #d4edda;
                    color: #155724;
                }
                
                .score-fail {
                    background-color: #f8d7da;
                    color: #721c24;
                }
                </style>
            """, unsafe_allow_html=True)

            # Display Code Assessment with enhanced citation context
            with st.expander("💻 Code Assessment", expanded=True):
                code_data = teaching_data.get("Code Assessment", {})
                
                for category, details in code_data.items():
                    score = details.get("Score", 0)
                    citations = details.get("Citations", [])
                    
                    st.markdown(f"""
                        <div class="teaching-card">
                            <div class="teaching-header">
                                <span class="category-name">{category}</span>
                                <span class="score-badge {'score-pass' if score == 1 else 'score-fail'}">
                                    {'✅ Pass' if score == 1 else '❌ Needs Work'}
                                </span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if citations:
                        st.markdown("##### 📝 Code Examples & Explanations")
                        for citation in citations:
                            # Extract timestamp if available
                            timestamp_match = re.match(r'\[(\d+:\d+)\]', citation)
                            timestamp = timestamp_match.group(1) if timestamp_match else "00:00"
                            
                            # Extract the actual quote
                            quote = re.sub(r'\[\d+:\d+\]\s*', '', citation).strip("'")
                            
                            # Find the quote in the transcript
                            transcript = evaluation.get("transcript", "")
                            if transcript and quote in transcript:
                                # Get surrounding context
                                quote_index = transcript.find(quote)
                                start_index = max(0, quote_index - 200)
                                end_index = min(len(transcript), quote_index + len(quote) + 200)
                                
                                context = transcript[start_index:end_index]
                                
                                if start_index > 0:
                                    context = "..." + context
                                if end_index < len(transcript):
                                    context = context + "..."
                                
                                highlighted_context = context.replace(quote, f'<span class="highlight-code">{quote}</span>')
                                
                                st.markdown(f"""
                                    <div class="code-citation-box">
                                        <div class="citation-timestamp">🕒 Timestamp: {timestamp}</div>
                                        <div class="code-citation-context">
                                            {highlighted_context}
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                    <div class="code-citation-box">
                                        <div class="citation-timestamp">🕒 Timestamp: {timestamp}</div>
                                        <div class="citation-text">"{quote}"</div>
                                    </div>
                                """, unsafe_allow_html=True)

            # Display Question Handling Assessment with enhanced citation context
            with st.expander("❓ Question Handling Assessment", expanded=True):
                question_data = concept_data.get("Question Handling", {})
                if question_data:
                    main_score = question_data.get("Score", 0)
                    main_citations = question_data.get("Citations", [])
                    details = question_data.get("Details", {})
                    
                    st.markdown(f"""
                        <div class="teaching-card">
                            <div class="teaching-header">
                                <span class="category-name">Overall Question Handling</span>
                                <span class="score-badge {'score-pass' if main_score == 1 else 'score-fail'}">
                                    {'✅ Pass' if main_score == 1 else '❌ Needs Work'}
                                </span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display main citations
                    if main_citations:
                        st.markdown("##### 📝 General Question Handling Examples")
                        for citation in main_citations:
                            # Process citation with context (same as above)
                            timestamp_match = re.match(r'\[(\d+:\d+)\]', citation)
                            timestamp = timestamp_match.group(1) if timestamp_match else "00:00"
                            quote = re.sub(r'\[\d+:\d+\]\s*', '', citation).strip("'")
                            
                            transcript = evaluation.get("transcript", "")
                            if transcript and quote in transcript:
                                quote_index = transcript.find(quote)
                                start_index = max(0, quote_index - 200)
                                end_index = min(len(transcript), quote_index + len(quote) + 200)
                                context = transcript[start_index:end_index]
                                
                                if start_index > 0:
                                    context = "..." + context
                                if end_index < len(transcript):
                                    context = context + "..."
                                
                                highlighted_context = context.replace(quote, f'<span class="highlight-question">{quote}</span>')
                                
                                st.markdown(f"""
                                    <div class="question-citation-box">
                                        <div class="citation-timestamp">🕒 Timestamp: {timestamp}</div>
                                        <div class="question-citation-context">
                                            {highlighted_context}
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                    
                    # Display detailed assessments
                    for aspect, aspect_details in details.items():
                        score = aspect_details.get("Score", 0)
                        citations = aspect_details.get("Citations", [])
                        
                        st.markdown(f"""
                            <div class="question-detail-card">
                                <div class="detail-header">
                                    <span class="detail-name">{aspect}</span>
                                    <span class="score-badge {'score-pass' if score == 1 else 'score-fail'}">
                                        {'✅ Pass' if score == 1 else '❌ Needs Work'}
                                    </span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        if citations:
                            for citation in citations:
                                # Process citation with context (same as above)
                                timestamp_match = re.match(r'\[(\d+:\d+)\]', citation)
                                timestamp = timestamp_match.group(1) if timestamp_match else "00:00"
                                quote = re.sub(r'\[\d+:\d+\]\s*', '', citation).strip("'")
                                
                                transcript = evaluation.get("transcript", "")
                                if transcript and quote in transcript:
                                    quote_index = transcript.find(quote)
                                    start_index = max(0, quote_index - 200)
                                    end_index = min(len(transcript), quote_index + len(quote) + 200)
                                    context = transcript[start_index:end_index]
                                    
                                    if start_index > 0:
                                        context = "..." + context
                                    if end_index < len(transcript):
                                        context = context + "..."
                                    
                                    highlighted_context = context.replace(quote, f'<span class="highlight-question">{quote}</span>')
                                    
                                    st.markdown(f"""
                                        <div class="question-detail-citation-box">
                                            <div class="citation-timestamp">🕒 Timestamp: {timestamp}</div>
                                            <div class="question-citation-context">
                                                {highlighted_context}
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)

            # Add additional CSS for code and question handling sections
            st.markdown("""
                <style>
                /* Code Assessment Styles */
                .code-citation-box {
                    background: #f8f9fa;
                    border-left: 4px solid #28a745;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 4px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    transition: transform 0.2s ease;
                }
                
                .code-citation-box:hover {
                    transform: translateX(5px);
                }
                
                .highlight-code {
                    background-color: #e8f5e9;
                    padding: 2px 4px;
                    border-radius: 2px;
                    font-weight: bold;
                    color: #2e7d32;
                }
                
                /* Question Handling Styles */
                .question-citation-box {
                    background: #f8f9fa;
                    border-left: 4px solid #9c27b0;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 4px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    transition: transform 0.2s ease;
                }
                
                .question-citation-box:hover {
                    transform: translateX(5px);
                }
                
                .highlight-question {
                    background-color: #f3e5f5;
                    padding: 2px 4px;
                    border-radius: 2px;
                    font-weight: bold;
                    color: #6a1b9a;
                }
                
                .question-detail-card {
                    background: #f8f9fa;
                    border-radius: 8px;
                    padding: 12px;
                    margin: 10px 0;
                    border-left: 4px solid #9c27b0;
                }
                
                .detail-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                
                .detail-name {
                    font-size: 1.1em;
                    font-weight: bold;
                    color: #4a148c;
                }
                
                .question-detail-citation-box {
                    background: white;
                    border-left: 3px solid #ce93d8;
                    padding: 12px;
                    margin: 10px 0;
                    border-radius: 4px;
                }
                </style>
            """, unsafe_allow_html=True)

        with tabs[2]:
            st.header("Recommendations")
            recommendations = evaluation.get("recommendations", {})
            
            # Add accent analysis section
            audio_features = evaluation.get("audio_features", {})
            accent_info = audio_features.get("accent_classification", {})
            
            if accent_info and accent_info.get("accent") != "Unknown":
                st.markdown("""
                    <div class="accent-card">
                        <h4>🗣️ Accent Analysis</h4>
                        <div class="accent-content">
                """, unsafe_allow_html=True)
                
                accent = accent_info.get("accent", "Unknown")
                confidence = accent_info.get("confidence", 0.0) * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Detected Accent", accent)
                with col2:
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Display summary in a styled card
            if "summary" in recommendations:
                st.markdown("""
                    <div class="summary-card">
                        <h4>📊 Overall Summary</h4>
                        <div class="summary-content">
                """, unsafe_allow_html=True)
                st.markdown(recommendations["summary"])
                st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Add Geography Fit and Rigor Assessment
            st.markdown("""
                <div class="assessment-card">
                    <h4>🌍 Teaching Assessment</h4>
                    <div class="assessment-content">
            """, unsafe_allow_html=True)
            
            # Geography Fit
            st.markdown("""
                <div class="assessment-item">
                    <h5>Geography Fit</h5>
                    <div class="assessment-text">
            """, unsafe_allow_html=True)
            geography_fit = recommendations.get("geographyFit", "Not available")
            st.markdown(f"<p>{geography_fit}</p>", unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Teaching Rigor
            st.markdown("""
                <div class="assessment-item">
                    <h5>Teaching Rigor</h5>
                    <div class="assessment-text">
            """, unsafe_allow_html=True)
            teaching_rigor = recommendations.get("rigor", "Not available")
            st.markdown(f"<p>{teaching_rigor}</p>", unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Update CSS for assessment items
            st.markdown("""
                <style>
                .assessment-card {
                    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                    padding: 20px;
                    border-radius: 8px;
                    margin: 15px 0;
                    border-left: 4px solid #4a69bd;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                }
                
                .assessment-content {
                    margin-top: 15px;
                }
                
                .assessment-item {
                    background: white;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px 0;
                    border: 1px solid #e9ecef;
                    transition: transform 0.2s ease;
                }
                
                .assessment-item:hover {
                    transform: translateX(5px);
                }
                
                .assessment-item h5 {
                    color: #4a69bd;
                    margin-bottom: 10px;
                    font-size: 1.1em;
                    border-bottom: 2px solid #f0f0f0;
                    padding-bottom: 5px;
                }
                
                .assessment-text {
                    color: #2c3e50;
                    font-size: 1em;
                    line-height: 1.6;
                    padding: 10px;
                    background: #f8f9fa;
                    border-radius: 4px;
                }
                
                .assessment-text p {
                    margin: 0;
                    padding: 0;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Add Profile Matches
            if "profileMatches" in recommendations:
                st.markdown("""
                    <div class="profiles-card">
                        <h4>👥 Learner Profile Matches</h4>
                        <div class="profiles-content">
                """, unsafe_allow_html=True)
                
                for profile in recommendations["profileMatches"]:
                    profile_name = profile.get("profile", "").replace("_", " ").title()
                    is_match = profile.get("match", False)
                    reason = profile.get("reason", "No reason provided")
                    
                    st.markdown(f"""
                        <div class="profile-item {'profile-match' if is_match else ''}">
                            <div class="profile-header">
                                <span class="profile-name">{profile_name}</span>
                                <span class="match-badge {'match-yes' if is_match else 'match-no'}">
                                    {'✅ Best Match' if is_match else '❌ Not Recommended'}
                                </span>
                            </div>
                            <div class="profile-reason">{reason}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Add Question Handling Assessment
            if "questionHandling" in recommendations:
                st.markdown("""
                    <div class="question-handling-card">
                        <h4>❓ Question Handling Assessment</h4>
                        <div class="question-content">
                """, unsafe_allow_html=True)
                
                q_handling = recommendations["questionHandling"]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence", q_handling.get("confidence", "Unknown"))
                with col2:
                    st.metric("Answer Accuracy", q_handling.get("accuracy", "Unknown"))
                
                if "improvements" in q_handling:
                    st.markdown("### Suggested Improvements")
                    for improvement in q_handling["improvements"]:
                        st.markdown(f"- {improvement}")
                
                st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Add Nervousness Analysis Section
            st.markdown("""
                <div class="nervousness-card">
                    <h4>😰 Nervousness Analysis</h4>
                    <div class="nervousness-content">
            """, unsafe_allow_html=True)
            
            # Get speech metrics
            speech_metrics = evaluation.get("speech_metrics", {})
            fluency_data = speech_metrics.get("fluency", {})
            flow_data = speech_metrics.get("flow", {})
            intonation_data = speech_metrics.get("intonation", {})
            energy_data = speech_metrics.get("energy", {})
            
            # Calculate nervousness indicators
            fillers_per_minute = float(fluency_data.get("fillersPerMin", 0))
            errors_per_minute = float(fluency_data.get("errorsPerMin", 0))
            pauses_per_minute = float(flow_data.get("pausesPerMin", 0))
            monotone_score = float(intonation_data.get("monotoneScore", 0))
            pitch_variation = float(intonation_data.get("pitchVariation", 0))
            mean_amplitude = float(energy_data.get("meanAmplitude", 0))
            
            # Create nervousness indicators
            nervousness_indicators = []
            
            if fillers_per_minute > 3:
                nervousness_indicators.append(f"High frequency of filler words ({fillers_per_minute:.1f}/min)")
            if errors_per_minute > 1:
                nervousness_indicators.append(f"Elevated speech errors ({errors_per_minute:.1f}/min)")
            if pauses_per_minute > 12:
                nervousness_indicators.append(f"Unnatural pauses ({pauses_per_minute:.1f}/min)")
            if monotone_score > 0.4:
                nervousness_indicators.append(f"Monotone speech pattern (score: {monotone_score:.2f})")
            if pitch_variation < 20 or pitch_variation > 40:
                nervousness_indicators.append(f"Unusual pitch variation ({pitch_variation:.1f}%)")
            if mean_amplitude < 60 or mean_amplitude > 75:
                nervousness_indicators.append(f"Voice energy level outside optimal range ({mean_amplitude:.1f})")
            
            if nervousness_indicators:
                st.warning("⚠️ Nervousness Indicators Detected:")
                for indicator in nervousness_indicators:
                    st.markdown(f"- {indicator}")
            else:
                st.success("✅ No significant nervousness indicators detected")
            
            st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Add Areas for Improvement section
            st.markdown("<h4>💡 Areas for Improvement</h4>", unsafe_allow_html=True)
            improvements = recommendations.get("improvements", [])
            
            if isinstance(improvements, list):
                # Use predefined categories
                categories = {
                    "🗣️ Communication": [],
                    "📚 Teaching": [],
                    "💻 Technical": []
                }
                
                # Each improvement should now come with a category from the content analysis
                for improvement in improvements:
                    if isinstance(improvement, dict):
                        category = improvement.get("category", "💻 Technical")  # Default to Technical if no category
                        message = improvement.get("message", str(improvement))
                        if "COMMUNICATION" in category.upper():
                            categories["🗣️ Communication"].append(message)
                        elif "TEACHING" in category.upper():
                            categories["📚 Teaching"].append(message)
                        elif "TECHNICAL" in category.upper():
                            categories["💻 Technical"].append(message)
                    else:
                        # Handle legacy format or plain strings
                        categories["💻 Technical"].append(improvement)
                
                # Display categorized improvements in columns
                cols = st.columns(len(categories))
                for col, (category, items) in zip(cols, categories.items()):
                    with col:
                        st.markdown(f"""
                            <div class="improvement-card">
                                <h5>{category}</h5>
                                <div class="improvement-list">
                        """, unsafe_allow_html=True)
                        
                        for item in items:
                            st.markdown(f"""
                                <div class="improvement-item">
                                    • {item}
                                </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Add additional CSS for new components
            st.markdown("""
                <style>
                .assessment-card, .profiles-card, .question-handling-card {
                    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                    padding: 20px;
                    border-radius: 8px;
                    margin: 15px 0;
                    border-left: 4px solid #4a69bd;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                }
                
                .profile-item {
                    background: white;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px 0;
                    border: 1px solid #e9ecef;
                    transition: transform 0.2s ease;
                }
                
                .profile-item:hover {
                    transform: translateX(5px);
                }
                
                .profile-match {
                    border-left: 4px solid #28a745;
                }
                
                .profile-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 10px;
                }
                
                .profile-name {
                    font-weight: bold;
                    color: #2c3e50;
                }
                
                .match-badge {
                    padding: 4px 8px;
                    border-radius: 15px;
                    font-size: 0.9em;
                }
                
                .match-yes {
                    background-color: #d4edda;
                    color: #155724;
                }
                
                .match-no {
                    background-color: #f8d7da;
                    color: #721c24;
                }
                
                .profile-reason {
                    color: #666;
                    font-size: 0.95em;
                    margin-top: 8px;
                    padding: 8px;
                    background: #f8f9fa;
                    border-radius: 4px;
                }
                
                .question-content {
                    margin-top: 15px;
                }
                </style>
            """, unsafe_allow_html=True)

        with tabs[3]:
            st.header("Transcript with Timestamps")
            transcript = evaluation.get("transcript", "")
            
            # Split transcript into sentences and add timestamps
            sentences = re.split(r'(?<=[.!?])\s+', transcript)
            for i, sentence in enumerate(sentences):
                # Calculate approximate timestamp based on words and average speaking rate
                words_before = len(' '.join(sentences[:i]).split())
                timestamp = words_before / 150  # Assuming 150 words per minute
                minutes = int(timestamp)
                seconds = int((timestamp - minutes) * 60)
                
                st.markdown(f"**[{minutes:02d}:{seconds:02d}]** {sentence}")

            # Comment out original transcript display
            # st.text(evaluation.get("transcript", "Transcript not available"))

    except Exception as e:
        logger.error(f"Error displaying evaluation: {e}")
        st.error(f"Error displaying results: {str(e)}")
        st.error("Please check the evaluation data structure and try again.")

    # Add these styles to the existing CSS in the main function
    st.markdown("""
        <style>
        /* ... existing styles ... */
        
        .citation-box {
            background-color: #f8f9fa;
            border-left: 3px solid #6c757d;
            padding: 10px;
            margin: 5px 0;
            border-radius: 0 4px 4px 0;
        }
        
        .recommendation-card {
            background-color: #ffffff;
            border-left: 4px solid #1f77b4;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .recommendation-card h4 {
            color: #1f77b4;
            margin: 0 0 10px 0;
        }
        
        .rigor-card {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            padding: 20px;
            margin: 10px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .score-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 15px;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .green-score {
            background-color: #28a745;
            color: white;
        }
        
        .orange-score {
            background-color: #fd7e14;
            color: white;
        }
        
        .metric-container {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        .profile-guide {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #1f77b4;
        }
        
        .profile-card {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        
        .profile-card.recommended {
            border-left: 4px solid #28a745;
        }
        
        .profile-header {
            margin-bottom: 15px;
        }
        
        .profile-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.9em;
            margin-top: 5px;
            background-color: #f8f9fa;
        }
        
        .profile-content ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        
        .recommendation-status {
            margin-top: 15px;
            padding: 10px;
            border-radius: 4px;
            background-color: #f8f9fa;
            font-weight: bold;
        }
        
        .recommendation-status small {
            display: block;
            margin-top: 5px;
            font-weight: normal;
            color: #666;
        }
        
        .recommendation-status.recommended {
            background-color: #d4edda;
            border-color: #c3e6cb;
            color: #155724;
        }
        
        .recommendation-status:not(.recommended) {
            background-color: #fff3cd;
            border-color: #ffeeba;
            color: #856404;
        }
        
        .profile-card.recommended {
            border-left: 4px solid #28a745;
            box-shadow: 0 2px 8px rgba(40, 167, 69, 0.1);
        }
        
        .profile-card:not(.recommended) {
            border-left: 4px solid #ffc107;
            opacity: 0.8;
        }
        
        .profile-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .progress-metric {
            background: linear-gradient(135deg, #f6f8fa 0%, #ffffff 100%);
            padding: 10px 15px;
            border-radius: 8px;
            border-left: 4px solid #1f77b4;
            margin: 5px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.2s ease;
        }
        
        .progress-metric:hover {
            transform: translateX(5px);
        }
        
        .progress-metric b {
            color: #1f77b4;
        }
        
        /* Enhanced status messages */
        .status-message {
            padding: 10px;
            border-radius: 8px;
            margin: 5px 0;
            animation: fadeIn 0.5s ease;
        }
        
        .status-processing {
            background: linear-gradient(135deg, #f0f7ff 0%, #e5f0ff 100%);
            border-left: 4px solid #1f77b4;
        }
        
        .status-complete {
            background: linear-gradient(135deg, #f0fff0 0%, #e5ffe5 100%);
            border-left: 4px solid #28a745;
        }
        
        .status-error {
            background: linear-gradient(135deg, #fff0f0 0%, #ffe5e5 100%);
            border-left: 4px solid #dc3545;
        }
        
        /* Progress bar enhancement */
        .stProgress > div > div {
            background-image: linear-gradient(
                to right,
                rgba(31, 119, 180, 0.8),
                rgba(31, 119, 180, 1)
            );
            transition: width 0.3s ease;
        }
        
        /* Batch indicator animation */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .batch-indicator {
            display: inline-block;
            padding: 4px 8px;
            background: #1f77b4;
            color: white;
            border-radius: 4px;
            animation: pulse 1s infinite;
        }
        
        .metric-box {
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            padding: 10px;
            border-radius: 8px;
            margin: 5px;
            border-left: 4px solid #1f77b4;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.2s ease;
        }
        
        .metric-box:hover {
            transform: translateX(5px);
        }
        
        .metric-box.batch {
            border-left-color: #28a745;
        }
        
        .metric-box.time {
            border-left-color: #dc3545;
        }
        
        .metric-box.progress {
            border-left-color: #ffc107;
        }
        
        .metric-box.segment {
            border-left-color: #17a2b8;
        }
        
        .metric-box b {
            color: #1f77b4;
        }
        
        <style>
        .metric-explanation-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            border-left: 4px solid #17a2b8;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .metric-explanation-card h4 {
            color: #17a2b8;
            margin-bottom: 10px;
        }
        
        .metric-explanation-card ul {
            list-style-type: none;
            padding-left: 0;
        }
        
        .metric-explanation-card li {
            margin-bottom: 12px;
            padding-left: 15px;
            border-left: 2px solid #e9ecef;
        }
        
        .metric-explanation-card li:hover {
            border-left: 2px solid #17a2b8;
        }
        </style>
        
        <style>
        /* ... existing styles ... */
        
        .suggestions-box {
            background-color: #f8f9fa;
            padding: 10px 15px;
            margin-top: 15px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
        }
        
        .suggestions-box h4 {
            color: #856404;
            margin: 0;
            padding: 5px 0;
        }
        
        .suggestion-item {
            padding: 5px 15px;
            color: #666;
            border-left: 2px solid #ffc107;
            margin: 5px 0;
            background-color: #fff;
            border-radius: 0 4px 4px 0;
        }
        
        .suggestion-item:hover {
            background-color: #fff9e6;
            transform: translateX(5px);
            transition: all 0.2s ease;
        }
        </style>
    """, unsafe_allow_html=True)
def check_dependencies() -> List[str]:
    """Check if required dependencies are installed"""
    missing = []
    
    if not shutil.which('ffmpeg'):
        missing.append("FFmpeg")
    
    return missing

def generate_pdf_report(evaluation_data: Dict[str, Any]) -> bytes:
    """Generate a more visually appealing and comprehensive PDF report."""
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                leftMargin=72, rightMargin=72,
                                topMargin=72, bottomMargin=72)
        styles = getSampleStyleSheet()
        story = []

        # --- Define Styles ---
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
                ('SPAN', (0, 0), (1, 0)), # Span title
                ('BACKGROUND', (0, 1), (1, 1), colors.HexColor('#4a69bd')), # Header background
                ('TEXTCOLOR', (0, 1), (1, 1), colors.whitesmoke),
                ('ALIGN', (0, 1), (1, 1), 'CENTER'),
                ('FONTNAME', (0, 1), (1, 1), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 1), (1, 1), 8),
                ('GRID', (0, 1), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                # Alternating row colors
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
        story.append(Paragraph("Mentor Demo Evaluation Report", title_style))
        story.append(Spacer(1, 20))

        # --- Communication Metrics ---
        story.append(Paragraph("Communication Metrics", heading1_style))
        speech_metrics = evaluation_data.get("speech_metrics", {})

        # Speed
        speed_data = speech_metrics.get("speed", {})
        story.extend(create_metric_table(
            "Speed", speed_data, ['wpm', 'total_words', 'duration_minutes']
        ))

        # Fluency
        fluency_data = speech_metrics.get("fluency", {})
        story.extend(create_metric_table(
             "Fluency", fluency_data, ['errorsPerMin', 'fillersPerMin']
        ))
        # Add detected fillers/errors details if available
        if fluency_data:
             fillers = fluency_data.get("detectedFillers", [])
             errors = fluency_data.get("detectedErrors", [])
             if fillers:
                  story.append(Paragraph("Detected Fillers:", heading2_style))
                  for f in fillers:
                       story.append(Paragraph(f"- {f.get('word', 'N/A')}: {f.get('count', 'N/A')}", body_style))
                  story.append(Spacer(1, 5))
             if errors:
                  story.append(Paragraph("Detected Errors:", heading2_style))
                  for e in errors:
                        story.append(Paragraph(f"- {e.get('type', 'N/A')} (Count: {e.get('count', 'N/A')}): {e.get('context', '')}", body_style))
                  story.append(Spacer(1, 10))


        # Flow
        flow_data = speech_metrics.get("flow", {})
        story.extend(create_metric_table("Flow", flow_data, ['pausesPerMin']))

        # Intonation & Energy (Combining audio features and speech metrics)
        audio_features = evaluation_data.get("audio_features", {})
        intonation_data = speech_metrics.get("intonation", {})
        energy_data = speech_metrics.get("energy", {})

        # Intonation Table
        intonation_display_data = {
             "Monotone Score": audio_features.get("monotone_score"),
             "Pitch Mean (Hz)": audio_features.get("pitch_mean"),
             "Pitch Variation Coeff (%)": audio_features.get("pitch_variation_coeff"),
             "Direction Changes/Min": audio_features.get("direction_changes_per_min"),
        }
        story.extend(create_metric_table("Intonation", intonation_display_data, list(intonation_display_data.keys())))

        # Energy Table
        energy_display_data = {
             "Mean Amplitude": audio_features.get("mean_amplitude"),
             "Amplitude Deviation": audio_features.get("amplitude_deviation"),
        }
        story.extend(create_metric_table("Energy", energy_display_data, list(energy_display_data.keys())))

        story.append(Spacer(1, 15))

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

        # --- Recommendations & Summary ---
        story.append(Paragraph("Recommendations & Summary", heading1_style))
        recommendations = evaluation_data.get("recommendations", {})

        # Summary
        summary = recommendations.get("summary", "N/A")
        story.extend(create_recommendation_section("Overall Summary", summary))

        # Improvements
        improvements = recommendations.get("improvements", [])
        story.extend(create_recommendation_section("Areas for Improvement", improvements))

        # Question Handling Specific Recommendations (if available)
        question_handling_rec = recommendations.get("questionHandling", {})
        if question_handling_rec:
             story.append(Paragraph("Question Handling Feedback:", heading2_style))
             q_confidence = question_handling_rec.get('confidence', 'N/A')
             q_accuracy = question_handling_rec.get('accuracy', 'N/A')
             q_improvements = question_handling_rec.get('improvements', [])
             story.append(Paragraph(f"<b>Confidence:</b> {q_confidence}", body_style))
             story.append(Paragraph(f"<b>Accuracy:</b> {q_accuracy}", body_style))
             if q_improvements:
                  story.append(Paragraph("<b>Improvements:</b>", body_style))
                  for imp in q_improvements:
                       story.append(Paragraph(f"- {imp}", body_style))
             story.append(Spacer(1, 15))


        # Rigor and Fit
        rigor = recommendations.get("rigor", "N/A")
        fit = recommendations.get("geographyFit", "N/A")
        story.append(Paragraph(f"Teaching Rigor Assessment: {rigor}", body_style))
        story.append(Paragraph(f"Geography Fit Assessment: {fit}", body_style))
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

def keep_device_active():
    """Keep the device active by periodically writing to the log"""
    try:
        while not st.session_state.get('processing_complete', False):  # Check processing status
            logger.info("Keeping device active...")
            time.sleep(30)  # Wait 30 seconds between logs
        logger.info("Processing complete, stopping device wake lock")
    except Exception as e:
        logger.warning(f"Device wake lock failed: {e}")

def clear_gpu_resources():
    """Clear GPU memory and turn off GPU after processing is complete."""
    try:
        # Clear PyTorch GPU cache if available
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Set device to CPU for subsequent operations
            torch.device('cpu')
            print("PyTorch GPU resources cleared")
            
        # Clear TensorFlow GPU memory if available
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                # Limit TensorFlow to CPU only after analysis
                tf.config.set_visible_devices([], 'GPU')
                print("TensorFlow GPU resources cleared")
        except ImportError:
            pass
            
        # Additional cleanup for other frameworks can be added here
            
    except Exception as e:
        print(f"Error clearing GPU resources: {e}")

def schedule_gpu_cleanup(delay_minutes=15):
    """
    Schedule GPU cleanup after specified delay in minutes.
    
    Args:
        delay_minutes: Number of minutes to wait before clearing GPU resources
    """
    import threading
    import time
    
    def delayed_cleanup():
        print(f"GPU cleanup scheduled to run in {delay_minutes} minutes")
        time.sleep(delay_minutes * 60)
        print("Executing scheduled GPU cleanup")
        clear_gpu_resources()
    
    # Start the cleanup in a background thread
    cleanup_thread = threading.Thread(target=delayed_cleanup)
    cleanup_thread.daemon = True  # Allow the thread to exit when the main program exits
    cleanup_thread.start()
    print(f"GPU cleanup scheduled for {delay_minutes} minutes from now")

def main():
    try:
        # Set page config must be the first Streamlit command
        st.set_page_config(page_title="🎓 Mentor Demo Review System", layout="wide")
        
        # Initialize session state for tracking progress
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        if 'evaluation_results' not in st.session_state:
            st.session_state.evaluation_results = None
        
        # Start device wake lock in background thread
        wake_thread = threading.Thread(target=keep_device_active, daemon=True)
        wake_thread.start()

        # Add custom CSS for animations and styling
        st.markdown("""
            <style>
                /* Shimmer animation keyframes */
                @keyframes shimmer {
                    0% {
                        background-position: -1000px 0;
                    }
                    100% {
                        background-position: 1000px 0;
                    }
                }
                
                .title-shimmer {
                    text-align: center;
                    color: #1f77b4;
                    position: relative;
                    overflow: hidden;
                    background: linear-gradient(
                        90deg,
                        rgba(255, 255, 255, 0) 0%,
                        rgba(255, 255, 255, 0.8) 50%,
                        rgba(255, 255, 255, 0) 100%
                    );
                    background-size: 1000px 100%;
                    animation: shimmer 3s infinite linear;
                }
                
                /* Existing animations */
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                
                @keyframes slideIn {
                    from { transform: translateX(-100%); }
                    to { transform: translateX(0); }
                }
                
                @keyframes pulse {
                    0% { transform: scale(1); }
                    50% { transform: scale(1.05); }
                    100% { transform: scale(1); }
                }
                
                .fade-in {
                    animation: fadeIn 1s ease-in;
                }
                
                .slide-in {
                    animation: slideIn 0.5s ease-out;
                }
                
                .pulse {
                    animation: pulse 2s infinite;
                }
                
                .metric-card {
                    background-color: #f0f2f6;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 10px 0;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    transition: transform 0.3s ease;
                }
                
                .metric-card:hover {
                    transform: translateY(-5px);
                }
                
                .stButton>button {
                    transition: all 0.3s ease;
                }
                
                .stButton>button:hover {
                    transform: scale(1.05);
                }
                
                .category-header {
                    background: linear-gradient(90deg, #1f77b4, #2c3e50);
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                }
                
                .score-badge {
                    padding: 5px 10px;
                    border-radius: 15px;
                    font-weight: bold;
                }
                
                .score-pass {
                    background-color: #28a745;
                    color: white;
                }
                
                .score-fail {
                    background-color: #dc3545;
                    color: white;
                }
                
                .metric-box {
                    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                    padding: 10px;
                    border-radius: 8px;
                    margin: 5px;
                    border-left: 4px solid #1f77b4;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    transition: transform 0.2s ease;
                }
                
                .metric-box:hover {
                    transform: translateX(5px);
                }
                
                .metric-box.batch {
                    border-left-color: #28a745;
                }
                
                .metric-box.time {
                    border-left-color: #dc3545;
                }
                
                .metric-box.progress {
                    border-left-color: #ffc107;
                }
                
                .metric-box.segment {
                    border-left-color: #17a2b8;
                }
                
                .metric-box b {
                    color: #1f77b4;
                }
            </style>
            
            <div class="fade-in">
                <h1 class="title-shimmer">
                    🎓 Mentor Demo Review System
                </h1>
            </div>
        """, unsafe_allow_html=True)

        # Sidebar with instructions and status
        with st.sidebar:
            st.markdown("""
                <div class="slide-in">
                    <h2>Instructions</h2>
                    <ol>
                        <li>Upload your teaching video</li>
                        <li>Wait for the analysis</li>
                        <li>Review the detailed feedback</li>
                        <li>Download the report</li>
                    </ol>
                </div>
            """, unsafe_allow_html=True)
            
            # Add file format information separately
            st.markdown("**Supported formats:** MP4, AVI, MOV")
            st.markdown("**Maximum file size:** 1GB")
            
            # Create a placeholder for status updates in the sidebar
            status_placeholder = st.empty()
            status_placeholder.info("Upload a video to begin analysis")

        # Check dependencies with progress
        with st.status("Checking system requirements...") as status:
            progress_bar = st.progress(0)
            
            status.update(label="Checking FFmpeg installation...")
            progress_bar.progress(0.3)
            missing_deps = check_dependencies()
            
            progress_bar.progress(0.6)
            if missing_deps:
                status.update(label="Missing dependencies detected!", state="error")
                st.error(f"Missing required dependencies: {', '.join(missing_deps)}")
                st.markdown("""
                Please install the missing dependencies:
                ```bash
                sudo apt-get update
                sudo apt-get install ffmpeg
                ```
                """)
                return
            
            progress_bar.progress(1.0)
            status.update(label="System requirements satisfied!", state="complete")

        # Add input selection with improved styling
        st.markdown("""
            <style>
            .input-selection {
                background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                border-left: 4px solid #1f77b4;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            
            .upload-section {
                background: #ffffff;
                padding: 20px;
                border-radius: 8px;
                margin-top: 15px;
                border: 1px solid #e0e0e0;
            }
            
            .upload-header {
                color: #1f77b4;
                font-size: 1.2em;
                margin-bottom: 10px;
            }
            </style>
        """, unsafe_allow_html=True)

        # Input type selection with better UI
        st.markdown('<div class="input-selection">', unsafe_allow_html=True)
        st.markdown("### 📤 Select Upload Method")
        input_type = st.radio(
            "Choose how you want to provide your teaching content:",
            options=[
                "Video Only (Auto-transcription)",
                "Video + Manual Transcript"
            ],
            help="Select whether you want to upload just the video (we'll transcribe it) or provide your own transcript"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Video upload section
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown('<p class="upload-header">📹 Upload Teaching Video</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Select video file",
            type=['mp4', 'avi', 'mov'],
            help="Upload your teaching video (MP4, AVI, or MOV format, max 1GB)"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Transcript upload section (conditional)
        uploaded_transcript = None
        if input_type == "Video + Manual Transcript":
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            st.markdown('<p class="upload-header">📝 Upload Transcript</p>', unsafe_allow_html=True)
            uploaded_transcript = st.file_uploader(
                "Select transcript file",
                type=['txt'],
                help="Upload your transcript (TXT format)"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Process video when uploaded
        if uploaded_file:
            if input_type == "Video + Manual Transcript" and not uploaded_transcript:
                st.warning("Please upload both video and transcript files to continue.")
                return
                
            # Only process if not already completed
            if not st.session_state.processing_complete:
                status_placeholder.info("Video uploaded, beginning processing...")
                
                # Add timer container
                timer_container = st.empty()
                start_time = time.time()
                
                st.markdown("""
                    <div class="pulse" style="text-align: center;">
                        <h3>Processing your video...</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Create temp directory for processing
                temp_dir = tempfile.mkdtemp()
                video_path = os.path.join(temp_dir, uploaded_file.name)
                
                try:
                    # Save uploaded file with progress
                    with st.status("Saving uploaded file...") as status:
                        # Update sidebar status
                        status_placeholder.info("Saving uploaded file...")
                        progress_bar = st.progress(0)
                        
                        # Save in chunks to show progress
                        chunk_size = 1024 * 1024  # 1MB chunks
                        file_size = len(uploaded_file.getbuffer())
                        chunks = file_size // chunk_size + 1
                        
                        with open(video_path, 'wb') as f:
                            for i in range(chunks):
                                # Update timer
                                elapsed_time = time.time() - start_time
                                minutes = int(elapsed_time // 60)
                                seconds = int(elapsed_time % 60)
                                timer_container.markdown(f"""
                                    <div style="
                                        background: linear-gradient(135deg, #f0f7ff 0%, #e5f0ff 100%);
                                        padding: 15px;
                                        border-radius: 8px;
                                        margin: 10px 0;
                                        border-left: 4px solid #1f77b4;
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                        <h3 style="margin:0;">⏱️ Processing Time</h3>
                                        <p style="font-size: 1.2em; margin: 10px 0;">
                                            Time elapsed: {minutes:02d}:{seconds:02d}
                                        </p>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                start = i * chunk_size
                                end = min(start + chunk_size, file_size)
                                f.write(uploaded_file.getbuffer()[start:end])
                                progress = (i + 1) / chunks
                                status.update(label=f"Saving file: {progress:.1%}")
                                progress_bar.progress(progress)
                        
                        status.update(label="File saved successfully!", state="complete")
                    
                    # Validate file size
                    file_size = os.path.getsize(video_path) / (1024 * 1024 * 1024)
                    if file_size > 1:
                        st.error("File size exceeds 1GB limit. Please upload a smaller file.")
                        return
                    
                    # Process video
                    status_placeholder.info("Processing video and generating analysis...")
                    
                    process_container = st.container()
                    with process_container:
                        st.markdown("""
                            <div class="processing-status">
                                <h3>🎥 Processing Video</h3>
                                <div class="status-details"></div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Create a background thread to update the timer
                        def update_timer(timer_container):
                            try:
                                while st.session_state.timer_running:
                                    if st.session_state.start_time is not None:
                                        current_time = time.time()
                                        elapsed_time = current_time - st.session_state.start_time
                                        minutes = int(elapsed_time // 60)
                                        seconds = int(elapsed_time % 60)
                                        
                                        timer_container.markdown(f"""
                                            <div style="
                                                background: linear-gradient(135deg, #f0f7ff 0%, #e5f0ff 100%);
                                                padding: 15px;
                                                border-radius: 8px;
                                                margin: 10px 0;
                                                border-left: 4px solid #1f77b4;
                                                box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                                <h3 style="margin:0;">⏱️ Processing Time</h3>
                                                <p style="font-size: 1.2em; margin: 10px 0;">
                                                    Time elapsed: {minutes:02d}:{seconds:02d}
                                                </p>
                                            </div>
                                        """, unsafe_allow_html=True)
                                    time.sleep(0.1)
                            except Exception as e:
                                logger.error(f"Timer update error: {e}")
                        
                        # Start timer update thread
                        timer_thread = threading.Thread(target=update_timer)
                        timer_thread.daemon = True
                        timer_thread.start()
                        
                        evaluator = MentorEvaluator()
                        st.session_state.evaluation_results = evaluator.evaluate_video(
                            video_path,
                            uploaded_transcript if input_type == "Video + Manual Transcript" else None
                        )
                        st.session_state.processing_complete = True
                        
                        # Final timer update
                        final_time = time.time() - start_time
                        minutes = int(final_time // 60)
                        seconds = int(final_time % 60)
                        timer_container.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, #f0fff0 0%, #e5ffe5 100%);
                                padding: 15px;
                                border-radius: 8px;
                                margin: 10px 0;
                                border-left: 4px solid #28a745;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                <h3 style="margin:0;">⏱️ Final Processing Time</h3>
                                <p style="font-size: 1.2em; margin: 10px 0;">
                                    Total time: {minutes:02d}:{seconds:02d}
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    status_placeholder.error(f"Error during processing: {str(e)}")
                    st.error(f"Error during evaluation: {str(e)}")
                    
                finally:
                    # Clean up temp files
                    if 'temp_dir' in locals():
                        shutil.rmtree(temp_dir)
            
            # Display results if processing is complete
            if st.session_state.processing_complete and st.session_state.evaluation_results:
                status_placeholder.success("Analysis complete! Review results below.")
                
                # Display processing time in a nice format
                processing_time = st.session_state.evaluation_results.get("processing_time", 0)
                minutes = int(processing_time // 60)
                seconds = int(processing_time % 60)
                
                # Add styled timer display
                st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                        padding: 15px;
                        border-radius: 8px;
                        margin: 10px 0;
                        border-left: 4px solid #1f77b4;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                        <h3 style="margin:0;">⏱️ Analysis Duration</h3>
                        <p style="font-size: 1.2em; margin: 10px 0;">
                            Total processing time: {} minutes and {} seconds
                        </p>
                    </div>
                """.format(minutes, seconds), unsafe_allow_html=True)
                
                st.success("Analysis complete!")
                display_evaluation(st.session_state.evaluation_results)
                
                # Add download options
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.download_button(
                        "📥 Download JSON Report",
                        json.dumps(st.session_state.evaluation_results, indent=2),
                        "evaluation_report.json",
                        "application/json",
                        help="Download the raw evaluation data in JSON format"
                    ):
                        st.success("JSON report downloaded successfully!")
                
                with col2:
                    if st.download_button(
                        "📄 Download Full Report (PDF)",
                        generate_pdf_report(st.session_state.evaluation_results),
                        "evaluation_report.pdf",
                        "application/pdf",
                        help="Download a formatted PDF report with detailed analysis"
                    ):
                        st.success("PDF report downloaded successfully!")

                # After displaying evaluation results, add:
                schedule_gpu_cleanup()
                st.success("Analysis completed and GPU resources released")

        # Add Hiring Recommendation Section
        if "hiringRecommendation" in recommendations:
            st.markdown("""
                <div class="hiring-card">
                    <h4>🎯 Hiring Recommendation</h4>
                    <div class="hiring-content">
            """, unsafe_allow_html=True)
            
            hiring_data = recommendations["hiringRecommendation"]
            score = hiring_data.get("score", 0)
            
            # Create a color gradient based on the score
            if score >= 8:
                score_color = "#28a745"  # Green for high scores
                recommendation = "Strongly Recommended"
            elif score >= 6:
                score_color = "#ffc107"  # Yellow for medium scores
                recommendation = "Recommended with Reservations"
            else:
                score_color = "#dc3545"  # Red for low scores
                recommendation = "Not Recommended"
            
            # Display the score with a circular progress indicator
            st.markdown(f"""
                <div style="text-align: center; margin: 20px 0;">
                    <div style="
                        width: 150px;
                        height: 150px;
                        border-radius: 50%;
                        border: 10px solid {score_color};
                        margin: 0 auto;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 48px;
                        font-weight: bold;
                        color: {score_color};">
                        {score}/10
                    </div>
                    <div style="
                        margin-top: 10px;
                        font-size: 1.2em;
                        font-weight: bold;
                        color: {score_color};">
                        {recommendation}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Display reasons in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ✅ Key Strengths")
                strengths = hiring_data.get("strengths", [])
                for strength in strengths:
                    st.markdown(f"- {strength}")
            
            with col2:
                st.markdown("### ⚠️ Areas of Concern")
                concerns = hiring_data.get("concerns", [])
                for concern in concerns:
                    st.markdown(f"- {concern}")
            
            # Display detailed reasons
            st.markdown("### 📝 Detailed Justification")
            reasons = hiring_data.get("reasons", [])
            for reason in reasons:
                st.markdown(f"""
                    <div class="reason-card">
                        • {reason}
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)

        # Add the new CSS styles for the hiring recommendation section
        st.markdown("""
            <style>
            .hiring-card {
                background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                padding: 20px;
                border-radius: 8px;
                margin: 15px 0;
                border-left: 4px solid #1f77b4;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            
            .hiring-content {
                margin-top: 15px;
            }
            
            .reason-card {
                background: white;
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                border-left: 3px solid #1f77b4;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                transition: transform 0.2s ease;
            }
            
            .reason-card:hover {
                transform: translateX(5px);
            }
            </style>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
    