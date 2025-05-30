from typing import Dict, List, Optional, Union
import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
import pandas as pd
from traffic_schema import TrafficEvent
import json

class TrafficNLPProcessor:
    def __init__(self):
        """Initialize the NLP processor with necessary models and templates"""
        # Load Slovenian spaCy model
        self.nlp = spacy.load("sl_core_news_lg")
        
        # Templates for different event types (can be learned from data)
        self.templates = {
            "wrong_way_driver": [
                "Opozarjamo vse voznike, ki vozijo po {road} od {from_loc} proti {to_loc}, "
                "da je na njihovo polovico {road_type} zašel voznik, ki vozi v napačno smer. "
                "Vozite skrajno desno in ne prehitevajte."
            ],
            "accident": [
                "Na {road} {section} proti {to_loc} je zaradi {reason} {consequence}.",
                "{reason} na {road} {section} proti {to_loc}. {consequence}."
            ]
        }
        
    def prepare_training_data(self, rtf_files: List[str], structured_events: List[Dict]) -> pd.DataFrame:
        """
        Prepare parallel data for training:
        - Input: Structured traffic events
        - Output: Natural language descriptions from RTF files
        """
        training_data = []
        
        for rtf, event in zip(rtf_files, structured_events):
            # Extract clean text from RTF
            clean_text = self._clean_rtf_text(rtf)
            
            # Create features from structured event
            features = self._extract_features(event)
            
            # Add to training data
            training_data.append({
                'input_features': features,
                'target_text': clean_text
            })
        
        return pd.DataFrame(training_data)
    
    def _extract_features(self, event: TrafficEvent) -> Dict:
        """Extract relevant features from traffic event for generation"""
        return {
            'event_type': event.event_type,
            'road_name': event.road_section.road_name,
            'direction': f"{event.road_section.direction.from_location}-{event.road_section.direction.to_location}",
            'section': event.road_section.direction.section,
            'reason': event.reason,
            'consequence': event.consequence,
            'priority': event.priority
        }
    
    def _clean_rtf_text(self, rtf_text: str) -> str:
        """Clean RTF text and extract main content"""
        # Remove RTF formatting
        text = rtf_text.replace('\\par', '\n')
        text = ' '.join(text.split())
        return text
    
    def analyze_report_structure(self, reports: List[str]):
        """Analyze linguistic patterns in existing reports"""
        patterns = []
        for report in reports:
            doc = self.nlp(report)
            
            # Analyze sentence structure
            for sent in doc.sents:
                pattern = []
                for token in sent:
                    # Get dependency pattern
                    pattern.append({
                        'dep': token.dep_,
                        'pos': token.pos_,
                        'text': token.text
                    })
                patterns.append(pattern)
        
        return self._extract_common_patterns(patterns)
    
    def _extract_common_patterns(self, patterns: List[List[Dict]]) -> List[Dict]:
        """Extract common linguistic patterns from reports"""
        common_patterns = {}
        
        for pattern in patterns:
            # Create pattern signature
            signature = tuple((p['dep'], p['pos']) for p in pattern)
            
            if signature not in common_patterns:
                common_patterns[signature] = {
                    'count': 0,
                    'examples': []
                }
            
            common_patterns[signature]['count'] += 1
            if len(common_patterns[signature]['examples']) < 3:
                common_patterns[signature]['examples'].append(
                    ' '.join(p['text'] for p in pattern)
                )
        
        # Sort by frequency
        return sorted(
            [{'pattern': k, **v} for k, v in common_patterns.items()],
            key=lambda x: x['count'],
            reverse=True
        )
    
    def train_generation_model(self, training_data: pd.DataFrame, model_name: str = "cjvt/t5-sl-small"):
        """Train a sequence-to-sequence model for report generation"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqGeneration.from_pretrained(model_name)
        
        # Prepare training examples
        train_encodings = self._prepare_training_examples(training_data)
        
        # Train the model (simplified - would need proper training loop)
        # This is where you'd implement the actual training
        pass
    
    def _prepare_training_examples(self, data: pd.DataFrame):
        """Prepare examples for training the generation model"""
        examples = []
        
        for _, row in data.iterrows():
            # Convert structured input to text format
            input_text = self._format_input_features(row['input_features'])
            target_text = row['target_text']
            
            # Tokenize
            encoded = self.tokenizer(
                input_text,
                target_text,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            examples.append(encoded)
            
        return examples
    
    def _format_input_features(self, features: Dict) -> str:
        """Format features into text input for the model"""
        return f"""dogodek: {features['event_type']}
                    cesta: {features['road_name']}
                    smer: {features['direction']}
                    odsek: {features['section']}
                    razlog: {features['reason']}
                    posledica: {features['consequence']}"""
    
    def generate_report(self, event: TrafficEvent) -> str:
        """Generate natural language report from traffic event"""
        # Extract features
        features = self._extract_features(event)
        
        # Format input
        input_text = self._format_input_features(features)
        
        # Generate report using the model
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)
        
        # Generate
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=150,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text

# Example usage:
"""
processor = TrafficNLPProcessor()

# 1. Prepare training data
rtf_files = [...] # Your RTF files
events = [...] # Your structured events
training_data = processor.prepare_training_data(rtf_files, events)

# 2. Analyze existing reports
patterns = processor.analyze_report_structure(rtf_files)
print("Common patterns found:", patterns[:3])

# 3. Train the model
processor.train_generation_model(training_data)

# 4. Generate new reports
event = TrafficEvent(...)
report = processor.generate_report(event)
print(report)
""" 