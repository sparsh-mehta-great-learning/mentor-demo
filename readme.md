# 🎓 Mentor Demo Review System

A sophisticated teaching evaluation system that analyzes video demos to provide comprehensive feedback on teaching effectiveness, communication skills, and technical content delivery.

## Features

### 🎯 Core Capabilities
- Video processing and audio extraction
- Automatic speech transcription
- Real-time analysis of teaching metrics
- Comprehensive evaluation across multiple dimensions
- PDF and JSON report generation

### 📊 Analysis Categories

#### Communication Metrics
- Speech speed and pacing
- Fluency and error detection
- Speech flow and pauses
- Voice intonation
- Energy levels and engagement

#### Teaching Assessment
- Subject matter accuracy
- First principles approach
- Examples and business context
- Cohesive storytelling
- Engagement and interaction
- Professional tone

#### Code Assessment
- Depth of explanation
- Output interpretation
- Breaking down complexity

## Requirements

### System Dependencies
- Python 3.7+
- FFmpeg
- Streamlit
- PyTorch
- OpenAI API access

### Python Dependencies
```bash
pip install streamlit torch numpy librosa whisper openai pandas soundfile faster-whisper reportlab
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mentor-demo-review
```

2. Install system dependencies:
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
export OPENAI_API_KEY='your-api-key'
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Access the web interface through your browser (typically http://localhost:8501)

3. Choose your input method:
   - Video Only (Auto-transcription)
   - Video + Manual Transcript

4. Upload your teaching video (supported formats: MP4, AVI, MOV)

5. Wait for the analysis to complete

6. Review the comprehensive evaluation results

7. Download reports in PDF or JSON format

## Features in Detail

### Audio Processing
- High-quality audio extraction
- Noise reduction and preprocessing
- Pause detection and analysis
- Voice pattern analysis

### Speech Analysis
- Words per minute calculation
- Filler word detection
- Grammatical error identification
- Intonation pattern analysis
- Energy level assessment

### Teaching Evaluation
- Content accuracy assessment
- Teaching methodology analysis
- Engagement measurement
- Professional tone evaluation
- Technical depth assessment

### Reporting
- Real-time progress tracking
- Interactive visualization
- Downloadable PDF reports
- Raw JSON data export
- Detailed citations and timestamps

## Technical Architecture

### Core Components
- `AudioFeatureExtractor`: Handles audio processing and feature extraction
- `ContentAnalyzer`: Analyzes teaching content using OpenAI API
- `RecommendationGenerator`: Generates teaching recommendations
- `MentorEvaluator`: Main evaluation orchestrator
- `ProgressTracker`: Manages analysis progress and UI updates

### Performance Features
- Efficient memory management
- Temporary file handling
- Multi-threading support
- Caching mechanisms
- Error handling and recovery

## Limitations

- Maximum file size: 1GB
- Supported video formats: MP4, AVI, MOV
- Requires stable internet connection for API calls
- Processing time varies with video length

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT API
- Whisper for speech recognition
- Streamlit for the web interface
- FFmpeg for video processing