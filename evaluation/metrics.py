"""
Evaluation metrics for code documentation generation
Includes BLEU, ROUGE, CodeBLEU, and human evaluation utilities
"""

import time
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import evaluate
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

class DocumentationEvaluator:
    """Comprehensive evaluation for code documentation generation"""
    
    def __init__(self):
        """Initialize evaluation metrics"""
        # Load metrics
        self.bleu = BLEU()
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
        # Try to load BERTScore (optional)
        try:
            self.bertscore = evaluate.load("bertscore")
        except:
            self.bertscore = None
            print("BERTScore not available. Install with: pip install bert-score")
    
    def compute_bleu(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Compute BLEU score"""
        # SacreBLEU expects list of references for each prediction
        refs = [[ref] for ref in references]
        
        # Compute corpus BLEU
        bleu_score = self.bleu.corpus_score(predictions, refs)
        
        return {
            'bleu': bleu_score.score,
            'bleu_1': bleu_score.precisions[0],
            'bleu_2': bleu_score.precisions[1],
            'bleu_3': bleu_score.precisions[2],
            'bleu_4': bleu_score.precisions[3]
        }
    
    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Compute ROUGE scores"""
        scores = defaultdict(list)
        
        for pred, ref in zip(predictions, references):
            score = self.rouge_scorer.score(ref, pred)
            
            for metric_name, metric_score in score.items():
                scores[f'{metric_name}_precision'].append(metric_score.precision)
                scores[f'{metric_name}_recall'].append(metric_score.recall)
                scores[f'{metric_name}_fmeasure'].append(metric_score.fmeasure)
        
        # Average scores
        avg_scores = {
            key: np.mean(values) * 100  # Convert to percentage
            for key, values in scores.items()
        }
        
        return avg_scores
    
    def compute_bertscore(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Compute BERTScore"""
        if self.bertscore is None:
            return {'bertscore_f1': 0.0}
        
        results = self.bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en"
        )
        
        return {
            'bertscore_precision': np.mean(results['precision']) * 100,
            'bertscore_recall': np.mean(results['recall']) * 100,
            'bertscore_f1': np.mean(results['f1']) * 100
        }
    
    def compute_code_bleu(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Simplified CodeBLEU approximation
        In practice, use: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans/evaluator/CodeBLEU
        """
        # For now, return regular BLEU as proxy
        # TODO: Implement full CodeBLEU with syntax matching
        bleu_scores = self.compute_bleu(predictions, references)
        return {
            'code_bleu': bleu_scores['bleu'] * 0.9  # Approximate adjustment
        }
    
    def compute_inference_time(
        self,
        model_generate_fn,
        test_samples: List[str],
        num_runs: int = 3
    ) -> Dict[str, float]:
        """Measure inference time and throughput"""
        times = []
        
        for _ in range(num_runs):
            start = time.time()
            for sample in test_samples:
                _ = model_generate_fn(sample)
            end = time.time()
            times.append(end - start)
        
        avg_time = np.mean(times)
        
        return {
            'total_time_seconds': avg_time,
            'time_per_sample_ms': (avg_time / len(test_samples)) * 1000,
            'throughput_samples_per_sec': len(test_samples) / avg_time
        }
    
    def evaluate_model(
        self,
        predictions: List[str],
        references: List[str],
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """Complete evaluation of a model"""
        print(f"\nEvaluating {model_name}...")
        print("="*60)
        
        results = {}
        
        # BLEU
        print("Computing BLEU...")
        bleu_scores = self.compute_bleu(predictions, references)
        results.update(bleu_scores)
        
        # ROUGE
        print("Computing ROUGE...")
        rouge_scores = self.compute_rouge(predictions, references)
        results.update(rouge_scores)
        
        # CodeBLEU
        print("Computing CodeBLEU...")
        codebleu_scores = self.compute_code_bleu(predictions, references)
        results.update(codebleu_scores)
        
        # BERTScore
        if self.bertscore:
            print("Computing BERTScore...")
            bertscore_scores = self.compute_bertscore(predictions, references)
            results.update(bertscore_scores)
        
        return results
    
    def compare_models(
        self,
        model_results: Dict[str, Dict[str, float]]
    ) -> None:
        """Print comparison table of multiple models"""
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        # Key metrics to compare
        key_metrics = [
            'bleu',
            'rouge1_fmeasure',
            'rouge2_fmeasure',
            'rougeL_fmeasure',
            'code_bleu',
            'bertscore_f1'
        ]
        
        # Print header
        print(f"{'Metric':<25}", end="")
        for model_name in model_results.keys():
            print(f"{model_name:<20}", end="")
        print()
        print("-"*80)
        
        # Print metrics
        for metric in key_metrics:
            print(f"{metric:<25}", end="")
            for model_name, results in model_results.items():
                value = results.get(metric, 0.0)
                print(f"{value:>18.2f}  ", end="")
            print()
        
        print("="*80)


class HumanEvaluationCollector:
    """Utilities for collecting human evaluation data"""
    
    CRITERIA = {
        'coherence': "Is the documentation coherent and well-structured?",
        'accuracy': "Does the documentation accurately describe the code?",
        'completeness': "Does the documentation cover all important aspects?",
        'usefulness': "Would this documentation be useful to a developer?"
    }
    
    def __init__(self, output_file: str = "human_eval_results.json"):
        self.output_file = output_file
        self.results = []
    
    def create_evaluation_form(
        self,
        code: str,
        generated_doc: str,
        reference_doc: str,
        sample_id: int
    ) -> Dict:
        """Create evaluation form for a single sample"""
        return {
            'sample_id': sample_id,
            'code': code,
            'generated_documentation': generated_doc,
            'reference_documentation': reference_doc,
            'criteria': self.CRITERIA,
            'scores': {criterion: None for criterion in self.CRITERIA.keys()},
            'comments': ""
        }
    
    def collect_sample_evaluation(
        self,
        sample: Dict,
        scores: Dict[str, int],
        comments: str = ""
    ) -> None:
        """Record evaluation for a sample"""
        sample['scores'] = scores
        sample['comments'] = comments
        self.results.append(sample)
    
    def save_results(self):
        """Save evaluation results"""
        import json
        with open(self.output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Human evaluation results saved to {self.output_file}")
    
    def compute_statistics(self) -> Dict[str, float]:
        """Compute statistics from human evaluation"""
        if not self.results:
            return {}
        
        stats = {}
        for criterion in self.CRITERIA.keys():
            scores = [r['scores'][criterion] for r in self.results if r['scores'][criterion]]
            stats[f'{criterion}_mean'] = np.mean(scores)
            stats[f'{criterion}_std'] = np.std(scores)
        
        return stats


def demo_evaluation():
    """Demo evaluation pipeline"""
    evaluator = DocumentationEvaluator()
    
    # Sample predictions and references
    predictions = [
        "This function calculates the factorial of a number using recursion.",
        "Sorts an array using the quicksort algorithm with pivot selection.",
        "Binary search implementation for finding elements in sorted arrays."
    ]
    
    references = [
        "Computes factorial of n recursively. Returns n! for positive integers.",
        "Implements quicksort algorithm. Sorts array in-place using divide and conquer.",
        "Binary search function. Finds target element in sorted array, returns index."
    ]
    
    # Evaluate
    results = evaluator.evaluate_model(predictions, references, "Demo Model")
    
    # Print results
    print("\nEvaluation Results:")
    print("-"*60)
    for metric, score in results.items():
        print(f"{metric:<30} {score:>10.2f}")
    
    return results


if __name__ == "__main__":
    demo_evaluation()
