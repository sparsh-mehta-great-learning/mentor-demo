#!/usr/bin/env python3
"""
Fix for Mentor Demo Review System Scoring Bug

This script demonstrates and fixes the issue where a good mentor (with Pass scores)
gets incorrectly evaluated as poor (2.0/10).
"""

def normalize_score(score_value):
    """Convert score to 1 if it indicates pass/success, 0 otherwise"""
    if isinstance(score_value, (int, float)):
        return 1 if score_value >= 0.5 else 0
    elif isinstance(score_value, str):
        return 1 if score_value.lower() in ['pass', 'excellent', 'good', 'yes', 'true'] else 0
    return 0

def calculate_hiring_score_fixed(metrics):
    """
    Fixed version of calculate_hiring_score that properly handles string values like "Pass"
    """
    # Get all required metrics
    speech_metrics = metrics.get("speech_metrics", {})
    teaching_data = metrics.get("teaching", {})
    concept_data = teaching_data.get("Concept Assessment", {})
    
    print("=== Debug: Checking input metrics ===")
    print(f"Speech metrics keys: {list(speech_metrics.keys())}")
    print(f"Teaching data keys: {list(teaching_data.keys())}")
    print(f"Concept data keys: {list(concept_data.keys())}")
    
    # 1. Communication Metrics (40% weight)
    comm_metrics = speech_metrics.get('intonation', {})
    speed_data = speech_metrics.get('speed', {})
    fluency_data = speech_metrics.get('fluency', {})
    
    # Extract WPM with proper fallback
    words_per_minute = speed_data.get('wpm', 0)
    print(f"Debug: Words per minute = {words_per_minute}")
    
    comm_scores = {
        'monotone': 1 if comm_metrics.get('monotone_score', 0) < 0.3 else 0,
        'pitch_variation': 1 if comm_metrics.get('pitchVariation', 0) >= 20 else 0,
        'direction_changes': 1 if comm_metrics.get('direction_changes_per_min', 0) >= 300 else 0,
        'fillers': 1 if fluency_data.get('fillersPerMin', 0) <= 3 else 0,
        'errors': 1 if fluency_data.get('errorsPerMin', 0) <= 1 else 0,
        'pace': 1 if 120 <= words_per_minute <= 180 else 0
    }
    
    print(f"Debug: Communication scores = {comm_scores}")
    comm_score = sum(comm_scores.values()) / len(comm_scores) * 4  # Convert to 0-4 scale
    print(f"Debug: Communication total score = {comm_score}/4")
    
    # 2. Teaching Metrics (40% weight) - Fixed to handle string values
    teaching_scores = {}
    for key, component in [
        ('content_accuracy', 'Subject Matter Accuracy'),
        ('examples', 'Examples and Business Context'),
        ('qna', 'Question Handling'),
        ('engagement', 'Engagement and Interaction')
    ]:
        raw_score = concept_data.get(component, {}).get('Score', 0)
        normalized = normalize_score(raw_score)
        teaching_scores[key] = normalized
        print(f"Debug: {component} = {raw_score} -> {normalized}")
    
    teaching_score = sum(teaching_scores.values()) / len(teaching_scores) * 4  # Convert to 0-4 scale
    print(f"Debug: Teaching total score = {teaching_score}/4")
    
    # 3. QnA Metrics (20% weight) - Fixed to handle string values  
    qna_data = concept_data.get('Question Handling', {})
    qna_details = qna_data.get('Details', {})
    
    qna_scores = {}
    for key, component in [
        ('response_accuracy', 'ResponseAccuracy'),
        ('response_completeness', 'ResponseCompleteness'),
        ('confidence', 'ConfidenceLevel')
    ]:
        raw_score = qna_details.get(component, {}).get('Score', 0)
        normalized = normalize_score(raw_score)
        qna_scores[key] = normalized
        print(f"Debug: QnA {component} = {raw_score} -> {normalized}")
    
    qna_score = sum(qna_scores.values()) / len(qna_scores) * 2  # Convert to 0-2 scale
    print(f"Debug: QnA total score = {qna_score}/2")
    
    # Calculate total score (0-10 scale)
    total_score = comm_score + teaching_score + qna_score
    print(f"Debug: Total score = {total_score}/10")
    
    # Generate assessment based on score
    if total_score >= 8:
        assessment = "Excellent"
        color = "#2ecc71"
        icon = "✅"
        description = "Outstanding performance across all metrics"
    elif total_score >= 6:
        assessment = "Good"
        color = "#f1c40f"
        icon = "⚠️"
        description = "Solid performance with some areas for improvement"
    elif total_score >= 4:
        assessment = "Needs Improvement"
        color = "#e67e22"
        icon = "⚠️"
        description = "Several areas need significant improvement"
    else:
        assessment = "Poor"
        color = "#e74c3c"
        icon = "❌"
        description = "Major improvements needed across multiple areas"
    
    return {
        "score": round(total_score, 1),
        "assessment": assessment,
        "color": color,
        "icon": icon,
        "description": description,
        "component_scores": {
            "communication": round(comm_score, 1),
            "teaching": round(teaching_score, 1),
            "qna": round(qna_score, 1)
        }
    }

def test_scoring_with_your_data():
    """Test the scoring with data similar to your report"""
    # Simulate the metrics from your report
    test_metrics = {
        "speech_metrics": {
            "speed": {
                "wpm": 178.8,  # This should be acceptable (120-180 range)
            },
            "fluency": {
                "fillersPerMin": 2.1,  # This is acceptable (<= 3)
                "errorsPerMin": 0.2,   # This is acceptable (<= 1)
            },
            "intonation": {
                "monotone_score": 0.03,        # This is good (< 0.3)
                "pitchVariation": 27.7,        # This is good (>= 20)
                "direction_changes_per_min": 392.76  # This is good (>= 300)
            }
        },
        "teaching": {
            "Concept Assessment": {
                "Subject Matter Accuracy": {"Score": "Pass"},
                "Examples and Business Context": {"Score": "Pass"},
                "Question Handling": {
                    "Score": "Pass",
                    "Details": {
                        "ResponseAccuracy": {"Score": "Pass"},
                        "ResponseCompleteness": {"Score": "Pass"},
                        "ConfidenceLevel": {"Score": "Pass"}
                    }
                },
                "Engagement and Interaction": {"Score": "Pass"}
            }
        }
    }
    
    print("=== Testing with your mentor's data ===")
    result = calculate_hiring_score_fixed(test_metrics)
    
    print(f"\n=== RESULTS ===")
    print(f"Overall Score: {result['score']}/10")
    print(f"Assessment: {result['assessment']}")
    print(f"Description: {result['description']}")
    print(f"Component Scores:")
    print(f"  - Communication: {result['component_scores']['communication']}/4")
    print(f"  - Teaching: {result['component_scores']['teaching']}/4")
    print(f"  - QnA: {result['component_scores']['qna']}/2")
    
    return result

if __name__ == "__main__":
    test_scoring_with_your_data() 