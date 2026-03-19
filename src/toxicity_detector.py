"""
Toxicity Detection Module
Detects hate speech, offensive language, and toxic comments
Uses a 3-class HateXplain checkpoint for classification
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List
from config import TOXICITY_MODEL, BATCH_SIZE


class ToxicityDetector:
    """
    Toxicity and Hate Speech Detector
    Uses Hate-speech-CNERG/bert-base-uncased-hatexplain
    """

    SUPPORTED_LABELS = {'hate', 'offensive', 'normal'}
    LABEL_ALIASES = {
        'hate': 'hate',
        'hate speech': 'hate',
        'hatespeech': 'hate',
        'offensive': 'offensive',
        'offensive language': 'offensive',
        'normal': 'normal',
        'not hate': 'normal',
        'not-hate': 'normal',
        'non hate': 'normal',
        'non-hate': 'normal',
        'neutral': 'normal',
        'neither': 'normal'
    }

    def __init__(self, model_name: str = None):
        self.model_name = model_name or TOXICITY_MODEL
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Loading toxicity model: {self.model_name}")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self._build_label_mapping()
        self.label2id = {label: idx for idx, label in self.id2label.items()}

    def _normalize_model_label(self, label: str) -> str:
        normalized = ' '.join(label.strip().lower().replace('_', ' ').replace('-', ' ').split())
        return self.LABEL_ALIASES.get(normalized, normalized)

    def _build_label_mapping(self) -> Dict[int, str]:
        config_labels = getattr(self.model.config, 'id2label', None) or {}
        normalized = {
            int(idx): self._normalize_model_label(label)
            for idx, label in config_labels.items()
        }

        if set(normalized.values()) != self.SUPPORTED_LABELS:
            raise ValueError(
                f"Unsupported toxicity label set for {self.model_name}: {normalized}"
            )

        return normalized

    def _default_result(self) -> Dict:
        return {
            'label': 'normal',
            'is_toxic': False,
            'is_hate': False,
            'is_offensive': False,
            'confidence': 0.0,
            'probabilities': {'hate': 0.0, 'offensive': 0.0, 'normal': 1.0}
        }

    def _format_probabilities(self, probs) -> Dict[str, float]:
        formatted = {'hate': 0.0, 'offensive': 0.0, 'normal': 0.0}
        for idx, score in enumerate(probs):
            formatted[self.id2label[idx]] = round(float(score), 4)
        return formatted

    def _build_result(self, predicted_class: int, confidence: float, probs) -> Dict:
        label = self.id2label[predicted_class]
        return {
            'label': label,
            'is_toxic': label in ['hate', 'offensive'],
            'is_hate': label == 'hate',
            'is_offensive': label == 'offensive',
            'confidence': round(float(confidence), 4),
            'probabilities': self._format_probabilities(probs)
        }

    def predict(self, text: str) -> Dict:
        """
        Predict toxicity for a single text

        Returns:
            Dictionary with toxicity label, confidence, and probabilities
        """
        if not text or not isinstance(text, str):
            return self._default_result()

        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        probs = probabilities[0].cpu().numpy()
        return self._build_result(predicted_class, confidence, probs)

    def predict_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> List[Dict]:
        """
        Predict toxicity for multiple texts in batches

        Args:
            texts: List of texts to analyze
            batch_size: Batch size for inference

        Returns:
            List of prediction dictionaries
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            valid_indices = [j for j, text in enumerate(batch) if text and isinstance(text, str)]
            valid_texts = [batch[j] for j in valid_indices]

            if not valid_texts:
                results.extend([self._default_result() for _ in batch])
                continue

            inputs = self.tokenizer(
                valid_texts,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()
                confidences = probabilities.max(dim=1)[0].cpu().numpy()
                all_probs = probabilities.cpu().numpy()

            batch_results = []
            valid_idx = 0
            for j in range(len(batch)):
                if j in valid_indices:
                    batch_results.append(
                        self._build_result(
                            int(predicted_classes[valid_idx]),
                            float(confidences[valid_idx]),
                            all_probs[valid_idx]
                        )
                    )
                    valid_idx += 1
                else:
                    batch_results.append(self._default_result())

            results.extend(batch_results)

        return results

    def analyze_comments(self, comments: List[Dict],
                        text_key: str = 'cleaned_text',
                        confidence_threshold: float = 0.7) -> List[Dict]:
        """
        Analyze toxicity for a list of comments

        Args:
            comments: List of comment dictionaries
            text_key: Key containing the text to analyze
            confidence_threshold: Minimum confidence to flag as toxic

        Returns:
            List of comments with added toxicity fields
        """
        texts = [comment.get(text_key, '') for comment in comments]
        predictions = self.predict_batch(texts)

        analyzed_comments = []
        for comment, prediction in zip(comments, predictions):
            comment_copy = comment.copy()
            comment_copy['toxicity'] = prediction['label']
            comment_copy['is_toxic'] = prediction['is_toxic'] and prediction['confidence'] >= confidence_threshold
            comment_copy['is_hate'] = prediction['is_hate'] and prediction['confidence'] >= confidence_threshold
            comment_copy['is_offensive'] = prediction['is_offensive'] and prediction['confidence'] >= confidence_threshold
            comment_copy['toxicity_confidence'] = prediction['confidence']
            comment_copy['toxicity_scores'] = prediction['probabilities']
            analyzed_comments.append(comment_copy)

        return analyzed_comments

    def get_toxicity_stats(self, comments: List[Dict]) -> Dict:
        """Get toxicity statistics"""
        total = len(comments)
        if total == 0:
            return {
                'total': 0,
                'toxic_count': 0,
                'hate_count': 0,
                'offensive_count': 0,
                'normal_count': 0,
                'toxicity_rate': 0.0
            }

        toxic = sum(1 for c in comments if c.get('is_toxic', False))
        hate = sum(1 for c in comments if c.get('is_hate', False))
        offensive = sum(1 for c in comments if c.get('is_offensive', False))
        normal = total - toxic

        return {
            'total': total,
            'toxic_count': toxic,
            'hate_count': hate,
            'offensive_count': offensive,
            'normal_count': normal,
            'toxicity_rate': round(toxic / total * 100, 2),
            'hate_rate': round(hate / total * 100, 2),
            'offensive_rate': round(offensive / total * 100, 2)
        }

    def get_toxic_comments(self, comments: List[Dict],
                          min_confidence: float = 0.7) -> List[Dict]:
        """Get list of toxic comments sorted by confidence"""
        toxic = [c for c in comments if c.get('is_toxic', False)
                 and c.get('toxicity_confidence', 0) >= min_confidence]
        return sorted(toxic, key=lambda x: x.get('toxicity_confidence', 0), reverse=True)
