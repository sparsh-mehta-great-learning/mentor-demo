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
        self.total_steps = 6  # Update total steps to include speech metrics
        self.substep_container = st.empty()
        self.metrics_container = st.container()
        
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
                "direction_changes_per_min": changes_per_minute
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
            "Citations": ["[MM:SS] Exact quote showing evidence"]
        }},
        "Professional Tone": {{
            "Score": 0 or 1,
            "Citations": ["[MM:SS] Exact quote showing evidence"]
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
- Encourages participation
- Shows audience awareness
- Uses engaging techniques
✗ Score 0 if MULTIPLE:
- Shows minimal interaction
- Ignores audience
- Lacks engagement attempts

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
"""

        return prompt_template.format(
            transcript=transcript,
            timestamp_instruction=timestamp_instruction
        )

    def _evaluate_speech_metrics(self, transcript: str, audio_features: Dict[str, float], 
                           progress_callback=None) -> Dict[str, Any]:
        """Evaluate speech metrics with improved accuracy"""
        try:
            if progress_callback:
                progress_callback(0.2, "Calculating speech metrics...")

            # Calculate words and duration
            words = len(transcript.split())
            duration_minutes = float(audio_features.get('duration', 0)) / 60
            words_per_minute = float(words / duration_minutes if duration_minutes > 0 else 0)
            
            # Calculate fluency metrics
            filler_words = ['um', 'uh', 'like', 'you know', 'sort of', 'kind of']
            filler_count = sum(transcript.lower().count(filler) for filler in filler_words)
            fillers_per_minute = float(filler_count / duration_minutes if duration_minutes > 0 else 0)
            
            # Detect speech errors (repetitions, incomplete sentences)
            words_list = transcript.split()
            repetitions = sum(1 for i in range(len(words_list)-1) if words_list[i] == words_list[i+1])
            incomplete_sentences = len(re.findall(r'[.!?]\s*[a-z]|[^.!?]$', transcript))
            total_errors = repetitions + incomplete_sentences
            errors_per_minute = float(total_errors / duration_minutes if duration_minutes > 0 else 0)
            
            # Basic speech metrics calculation
            return {
                "speed": {
                    "score": 1 if 120 <= words_per_minute <= 180 else 0,
                    "wpm": words_per_minute,
                    "total_words": words,
                    "duration_minutes": duration_minutes
                },
                "fluency": {
                    "score": 1 if fillers_per_minute <= 3 and errors_per_minute <= 1 else 0,
                    "errorsPerMin": errors_per_minute,
                    "fillersPerMin": fillers_per_minute,
                    "maxErrorsThreshold": 1.0,
                    "maxFillersThreshold": 3.0,
                    "details": {
                        "filler_count": filler_count,
                        "repetitions": repetitions,
                        "incomplete_sentences": incomplete_sentences
                    }
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
                    "variationsPerMin": audio_features.get("variations_per_minute", 0)
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
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
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
    """Display evaluation results with improved metrics visualization"""
    try:
        tabs = st.tabs(["Communication", "Teaching", "Recommendations", "Transcript"])
        
        with tabs[0]:
            st.header("Communication Metrics")
            
            # Get audio features and ensure we have the required metrics
            audio_features = evaluation.get("audio_features", {})
            
            # Speed Metrics
            with st.expander("🏃 Speed", expanded=True):
                # Fix: Calculate WPM using total words and duration
                speech_metrics = evaluation.get("speech_metrics", {})
                speed_data = speech_metrics.get("speed", {})
                words_per_minute = speed_data.get("wpm", 0)  # Get WPM from speech metrics
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Score", "✅ Pass" if 120 <= words_per_minute <= 180 else "❌ Needs Improvement")
                    st.metric("Words per Minute", f"{words_per_minute:.1f}")
                with col2:
                    st.info("""
                    **Acceptable Range:** 120-180 WPM
                    - Optimal teaching pace: 130-160 WPM
                    """)

            # Fluency Metrics
            with st.expander("🗣️ Fluency", expanded=True):
                # Get metrics from speech evaluation
                speech_metrics = evaluation.get("speech_metrics", {})
                fillers_per_minute = float(speech_metrics.get("fluency", {}).get("fillersPerMin", 0))
                errors_per_minute = float(speech_metrics.get("fluency", {}).get("errorsPerMin", 0))
                
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
                    """)

            # Flow Metrics
            with st.expander("🌊 Flow", expanded=True):
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
                    
                    # Add explanation card
                    st.markdown("""
                    <div class="metric-explanation-card">
                        <h4>📊 Understanding Flow Metrics</h4>
                        <ul>
                            <li><strong>Pauses per Minute (PPM):</strong> Measures the frequency of natural breaks in speech. Strategic pauses help learners process information and emphasize key points.</li>
                            <li><strong>Optimal Range:</strong> 8-12 PPM indicates well-paced delivery with appropriate breaks for comprehension.</li>
                            <li><strong>Impact:</strong> Too few pauses can overwhelm learners, while too many can disrupt flow and engagement.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

            # Intonation Metrics
            with st.expander("🎵 Intonation", expanded=True):
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
            with st.expander("⚡ Energy", expanded=True):
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
            
            # Display Concept Assessment with AI-generated suggestions
            with st.expander("📚 Concept Assessment", expanded=True):
                concept_data = teaching_data.get("Concept Assessment", {})
                
                for category, details in concept_data.items():
                    score = details.get("Score", 0)
                    citations = details.get("Citations", [])
                    
                    # Get AI-generated suggestions if score is 0
                    suggestions = []
                    if score == 0:
                        suggestions = content_analyzer.generate_suggestions(category, citations)
                    
                    # Create suggestions based on score and category
                    st.markdown(f"""
                        <div class="teaching-card">
                            <div class="teaching-header">
                                <span class="category-name">{category}</span>
                                <span class="score-badge {'score-pass' if score == 1 else 'score-fail'}">
                                    {'✅ Pass' if score == 1 else '❌ Needs Work'}
                                </span>
                            </div>
                            <div class="citations-container">
                    """, unsafe_allow_html=True)
                    
                    # Display citations
                    for citation in citations:
                        st.markdown(f"""
                            <div class="citation-box">
                                <i class="citation-text">{citation}</i>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Display AI-generated suggestions if score is 0
                    if score == 0 and suggestions:
                        st.markdown("""
                            <div class="suggestions-box">
                                <h4>🎯 Suggestions for Improvement:</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        for suggestion in suggestions:
                            st.markdown(f"""
                                <div class="suggestion-item">
                                    • {suggestion}
                                </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("</div></div>", unsafe_allow_html=True)
                    st.markdown("---")
            
            # Display Code Assessment with AI-generated suggestions
            with st.expander("💻 Code Assessment", expanded=True):
                code_data = teaching_data.get("Code Assessment", {})
                
                for category, details in code_data.items():
                    score = details.get("Score", 0)
                    citations = details.get("Citations", [])
                    
                    # Get AI-generated suggestions if score is 0
                    suggestions = []
                    if score == 0:
                        suggestions = content_analyzer.generate_suggestions(category, citations)
                    
                    # Create suggestions based on score and category
                    st.markdown(f"""
                        <div class="teaching-card">
                            <div class="teaching-header">
                                <span class="category-name">{category}</span>
                                <span class="score-badge {'score-pass' if score == 1 else 'score-fail'}">
                                    {'✅ Pass' if score == 1 else '❌ Needs Work'}
                                </span>
                            </div>
                            <div class="citations-container">
                    """, unsafe_allow_html=True)
                    
                    for citation in citations:
                        st.markdown(f"""
                            <div class="citation-box">
                                <i class="citation-text">{citation}</i>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Display AI-generated suggestions if score is 0
                    if score == 0 and suggestions:
                        st.markdown("""
                            <div class="suggestions-box">
                                <h4>🎯Suggestions for Improvement:</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        for suggestion in suggestions:
                            st.markdown(f"""
                                <div class="suggestion-item">
                                    • {suggestion}
                                </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("</div></div>", unsafe_allow_html=True)
                    st.markdown("---")

        with tabs[2]:
            st.header("Recommendations")
            recommendations = evaluation.get("recommendations", {})
            
            # Display summary in a styled card
            if "summary" in recommendations:
                st.markdown("""
                    <div class="summary-card">
                        <h4>📊 Overall Assessment</h4>
                        <div class="summary-content">
                            {}
                        </div>
                    </div>
                """.format(recommendations["summary"]), unsafe_allow_html=True)
            
            # Display improvements in categorized columns
            st.markdown("<h4>💡 Areas for Improvement</h4>", unsafe_allow_html=True)
            improvements = recommendations.get("improvements", [])
            
            # Initialize category buckets
            categorized_improvements = {
                "Communication": [],
                "Teaching": [],
                "Technical": []
            }
            
            # Sort improvements into categories
            for improvement in improvements:
                if isinstance(improvement, dict):
                    category = improvement.get("category", "").upper()
                    message = improvement.get("message", "")
                    
                    if "COMMUNICATION" in category:
                        categorized_improvements["Communication"].append(message)
                    elif "TEACHING" in category:
                        categorized_improvements["Teaching"].append(message)
                    elif "TECHNICAL" in category:
                        categorized_improvements["Technical"].append(message)
                else:
                    # Handle string improvements (legacy format)
                    categorized_improvements["Technical"].append(str(improvement))
            
            # Create columns for each category
            cols = st.columns(3)
            
            # Display improvements in columns with icons
            for col, (category, items) in zip(cols, categorized_improvements.items()):
                with col:
                    icon = "🗣️" if category == "Communication" else "📚" if category == "Teaching" else "💻"
                    st.markdown(f"""
                        <div class="improvement-card">
                            <h5>{icon} {category}</h5>
                            <div class="improvement-list">
                        """, unsafe_allow_html=True)
                    
                    if items:
                        for item in items:
                            st.markdown(f"""
                                <div class="improvement-item">
                                    • {item}
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class="improvement-item no-improvements">
                                No specific improvements needed in this category.
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Add additional CSS for recommendations styling
            st.markdown("""
                <style>
                .summary-card {
                    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                    border-radius: 8px;
                    padding: 20px;
                    margin: 15px 0;
                    border-left: 4px solid #1f77b4;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                
                .summary-card h4 {
                    color: #1f77b4;
                    margin-bottom: 15px;
                }
                
                .summary-content {
                    color: #495057;
                    line-height: 1.6;
                }
                
                .improvement-card {
                    background: white;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px 0;
                    height: 100%;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    border-left: 4px solid #28a745;
                }
                
                .improvement-card h5 {
                    color: #1f77b4;
                    margin-bottom: 15px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid #f0f0f0;
                }
                
                .improvement-list {
                    margin-top: 10px;
                }
                
                .improvement-item {
                    padding: 8px;
                    margin: 5px 0;
                    background: #f8f9fa;
                    border-radius: 4px;
                    color: #495057;
                    transition: transform 0.2s ease;
                }
                
                .improvement-item:hover {
                    transform: translateX(5px);
                    background: #f0f0f0;
                }
                
                .no-improvements {
                    color: #6c757d;
                    font-style: italic;
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
    """Generate a formatted PDF report from evaluation data"""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from io import BytesIO
        
        # Create PDF buffer
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        story.append(Paragraph("Mentor Demo Evaluation Report", title_style))
        story.append(Spacer(1, 20))
        
        # Communication Metrics Section
        story.append(Paragraph("Communication Metrics", styles['Heading2']))
        comm_metrics = evaluation_data.get("communication", {})
        
        # Create tables for each metric category
        for category in ["speed", "fluency", "flow", "intonation", "energy"]:
            if category in comm_metrics:
                metrics = comm_metrics[category]
                story.append(Paragraph(category.title(), styles['Heading3']))
                
                data = [[k.replace('_', ' ').title(), str(v)] for k, v in metrics.items()]
                t = Table(data, colWidths=[200, 200])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(t)
                story.append(Spacer(1, 20))
        
        # Teaching Analysis Section
        story.append(Paragraph("Teaching Analysis", styles['Heading2']))
        teaching_data = evaluation_data.get("teaching", {})
        
        for assessment_type in ["Concept Assessment", "Code Assessment"]:
            if assessment_type in teaching_data:
                story.append(Paragraph(assessment_type, styles['Heading3']))
                categories = teaching_data[assessment_type]
                
                for category, details in categories.items():
                    score = details.get("Score", 0)
                    citations = details.get("Citations", [])
                    
                    data = [
                        [category, "Score: " + ("Pass" if score == 1 else "Needs Improvement")],
                        ["Citations:", ""]
                    ] + [["-", citation] for citation in citations]
                    
                    t = Table(data, colWidths=[200, 300])
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 20))
        
        # Recommendations Section
        story.append(Paragraph("Recommendations", styles['Heading2']))
        recommendations = evaluation_data.get("recommendations", {})
        
        if "summary" in recommendations:
            story.append(Paragraph("Overall Summary:", styles['Heading3']))
            story.append(Paragraph(recommendations["summary"], styles['Normal']))
            story.append(Spacer(1, 20))
        
        if "improvements" in recommendations:
            story.append(Paragraph("Areas for Improvement:", styles['Heading3']))
            improvements = recommendations["improvements"]
            for improvement in improvements:
                # Handle both string and dictionary improvement formats
                if isinstance(improvement, dict):
                    message = improvement.get("message", "")
                    category = improvement.get("category", "")
                    story.append(Paragraph(f"• [{category}] {message}", styles['Normal']))
                else:
                    story.append(Paragraph(f"• {improvement}", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        pdf_data = buffer.getvalue()
        buffer.close()
        
        return pdf_data
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        raise RuntimeError(f"Failed to generate PDF report: {str(e)}")

def main():
    try:
        # Set page config must be the first Streamlit command
        st.set_page_config(page_title="🎓 Mentor Demo Review System", layout="wide")
        
        # Initialize session state for tracking progress
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        if 'evaluation_results' not in st.session_state:
            st.session_state.evaluation_results = None
        
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
                        
                        evaluator = MentorEvaluator()
                        
                        # Read transcript content if provided
                        transcript_content = None
                        if uploaded_transcript:
                            transcript_content = uploaded_transcript.getvalue().decode('utf-8')
                        
                        st.session_state.evaluation_results = evaluator.evaluate_video(
                            video_path,
                            transcript_content  # Pass the transcript content instead of the file object
                        )
                        st.session_state.processing_complete = True
                        
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

    except Exception as e:
        st.error(f"Application error: {str(e)}")

class MentorEvaluator:
    """Coordinates the evaluation process for mentor demos"""
    def __init__(self):
        self.audio_extractor = AudioFeatureExtractor()
        self.content_analyzer = ContentAnalyzer(st.secrets["OPENAI_API_KEY"])
        
    def evaluate_video(self, video_path: str, transcript_content: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a teaching video and generate comprehensive analysis
        """
        try:
            # Create progress tracking
            status_container = st.empty()
            progress_bar = st.progress(0)
            progress = ProgressTracker(status_container, progress_bar)
            
            # Create a temporary directory that will persist throughout the function
            with tempfile.TemporaryDirectory() as temp_dir:
                # Step 1: Extract audio from video
                progress.update(0.0, "Extracting audio from video...")
                audio_path = os.path.join(temp_dir, 'audio.wav')
                
                try:
                    subprocess.run([
                        'ffmpeg', '-i', video_path,
                        '-vn', '-acodec', 'pcm_s16le',
                        '-ar', '16000', '-ac', '1',
                        audio_path
                    ], check=True, capture_output=True)
                except subprocess.SubprocessError as e:
                    logger.error(f"FFmpeg error: {e}")
                    raise AudioProcessingError(f"Failed to process video audio: {str(e)}")
                
                progress.next_step()
                
                # Step 2: Generate transcript if not provided
                progress.update(0.0, "Processing audio...")
                if transcript_content:
                    transcript = transcript_content
                else:
                    # Initialize Whisper model
                    model = WhisperModel("base", device="cpu", compute_type="int8")
                    segments, _ = model.transcribe(audio_path, beam_size=5)
                    transcript = " ".join([segment.text for segment in segments])
                progress.next_step()
                
                # Step 3: Extract audio features
                progress.update(0.0, "Analyzing audio features...")
                # Verify file exists before processing
                if not os.path.exists(audio_path):
                    raise AudioProcessingError(f"Audio file not found at {audio_path}")
                    
                audio_features = self.audio_extractor.extract_features(
                    audio_path,
                    progress_callback=lambda p, m: progress.update(p, "Analyzing audio features...", m)
                )
                progress.next_step()
                
                # Step 4: Calculate speech metrics (Add this step)
                progress.update(0.0, "Analyzing speech patterns...")
                speech_metrics = self._evaluate_speech_metrics(
                    transcript,
                    audio_features,
                    progress_callback=lambda p, m: progress.update(p, "Analyzing speech patterns...", m)
                )
                progress.next_step()
                
                # Step 5: Analyze teaching content
                progress.update(0.0, "Analyzing teaching content...")
                teaching_analysis = self.content_analyzer.analyze_content(
                    transcript,
                    progress_callback=lambda p, m: progress.update(p, "Analyzing teaching content...", m)
                )
                progress.next_step()
                
                # Step 6: Generate final evaluation
                progress.update(0.0, "Generating final evaluation...")
                evaluation = {
                    "audio_features": audio_features,
                    "speech_metrics": speech_metrics,  # Include speech metrics in the evaluation
                    "transcript": transcript,
                    "teaching": teaching_analysis,
                    "recommendations": self._generate_recommendations(audio_features, teaching_analysis)
                }
                progress.next_step()
                
                return evaluation
                
        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            raise

    def _generate_recommendations(self, audio_features: Dict[str, float], 
                                teaching_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on analysis results"""
        recommendations = {
            "summary": "",
            "improvements": []
        }
        
        try:
            # Generate summary and improvements using GPT-4
            analysis_prompt = f"""
            Based on the following teaching analysis and audio metrics, provide:
            1. A brief summary of the teaching performance
            2. Specific areas for improvement with actionable suggestions
            
            Audio Metrics:
            {json.dumps(audio_features, indent=2)}
            
            Teaching Analysis:
            {json.dumps(teaching_analysis, indent=2)}
            
            Format response as JSON:
            {{
                "summary": "brief overall assessment",
                "improvements": [
                    {{"category": "COMMUNICATION/TEACHING/TECHNICAL", "message": "specific suggestion"}}
                ]
            }}
            """
            
            response = self.content_analyzer.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a teaching evaluation expert providing constructive feedback."},
                    {"role": "user", "content": analysis_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            recommendations = json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations["summary"] = "Error generating detailed recommendations."
            recommendations["improvements"] = [
                {"category": "TECHNICAL", "message": "Unable to generate specific recommendations."}
            ]
        
        return recommendations

    def _evaluate_speech_metrics(self, transcript: str, audio_features: Dict[str, float], 
                           progress_callback=None) -> Dict[str, Any]:
        """Evaluate speech metrics with improved accuracy"""
        try:
            if progress_callback:
                progress_callback(0.2, "Calculating speech metrics...")

            # Calculate words and duration
            words = len(transcript.split())
            duration_minutes = float(audio_features.get('duration', 0)) / 60
            words_per_minute = float(words / duration_minutes if duration_minutes > 0 else 0)
            
            # Calculate fluency metrics
            filler_words = ['um', 'uh', 'like', 'you know', 'sort of', 'kind of']
            filler_count = sum(transcript.lower().count(filler) for filler in filler_words)
            fillers_per_minute = float(filler_count / duration_minutes if duration_minutes > 0 else 0)
            
            # Detect speech errors (repetitions, incomplete sentences)
            words_list = transcript.split()
            repetitions = sum(1 for i in range(len(words_list)-1) if words_list[i] == words_list[i+1])
            incomplete_sentences = len(re.findall(r'[.!?]\s*[a-z]|[^.!?]$', transcript))
            total_errors = repetitions + incomplete_sentences
            errors_per_minute = float(total_errors / duration_minutes if duration_minutes > 0 else 0)
            
            # Basic speech metrics calculation
            return {
                "speed": {
                    "score": 1 if 120 <= words_per_minute <= 180 else 0,
                    "wpm": words_per_minute,
                    "total_words": words,
                    "duration_minutes": duration_minutes
                },
                "fluency": {
                    "score": 1 if fillers_per_minute <= 3 and errors_per_minute <= 1 else 0,
                    "errorsPerMin": errors_per_minute,
                    "fillersPerMin": fillers_per_minute,
                    "maxErrorsThreshold": 1.0,
                    "maxFillersThreshold": 3.0,
                    "details": {
                        "filler_count": filler_count,
                        "repetitions": repetitions,
                        "incomplete_sentences": incomplete_sentences
                    }
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
                    "variationsPerMin": audio_features.get("variations_per_minute", 0)
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

if __name__ == "__main__":
    main()