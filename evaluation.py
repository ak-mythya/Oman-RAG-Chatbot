import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import warnings
import os
import re
from collections import defaultdict
import asyncio
import time
from google import genai
warnings.filterwarnings('ignore')

# LLM Client imports (configure as needed)
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("OpenAI not installed. Install with: pip install openai")

# RAGAS imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_precision,
        context_recall,
        context_relevancy,
        answer_correctness,
        answer_similarity
    )
    from datasets import Dataset
    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False
    print("RAGAS not installed. Install with: pip install ragas")

@dataclass
class MultiLevelEvaluationResults:
    """Container for storing multi-level evaluation results"""
    step_name: str
    pooled_metrics: Dict[str, float]
    sub_query_metrics: Dict[str, float]
    synthesis_metrics: Dict[str, float]
    additional_info: Dict[str, Any]

class LLMJudge:
    """LLM-as-a-Judge implementation for hybrid evaluation"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4", max_retries: int = 3):
        self.model = model
        self.max_retries = max_retries
        self.client = genai.Client(api_key=api_key)
    
    def _call_llm(self, prompt: str, temperature: float = 0) -> str:
        if not self.client:
            return ""

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    contents=prompt,
                    model="gemini-2.0-flash"
                )
                return response.text.strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"LLM call failed after {self.max_retries} attempts: {e}")
                    return ""
                time.sleep(2 ** attempt)  # Exponential backoff
        return ""
    
    def evaluate_subquery_decomposition(self, original_query: str, sub_queries: List[str]) -> Dict[str, float]:
        """Evaluate sub-query decomposition quality"""
        prompt = f"""
        Evaluate how well the original query has been decomposed into sub-queries.
        
        Original Query: {original_query}
        
        Sub-queries:
        {chr(10).join([f"- {q}" for q in sub_queries])}
        
        Rate on these criteria (1-5 scale):
        1. Coverage: Do the sub-queries cover all important aspects of the original query?
        2. Non-redundancy: Are there minimal overlaps between sub-queries?
        3. Quality: Are the sub-queries well-formed and clear?
        4. Logical_decomposition: Is this a logical way to break down the original query?
        
        Respond in this exact format:
        Coverage: X
        Non_redundancy: X
        Quality: X
        Logical_decomposition: X
        Overall: X
        """
        
        response = self._call_llm(prompt)
        scores = {}
        
        for line in response.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                try:
                    scores[key] = float(value.strip())
                except ValueError:
                    continue
        
        return scores
    
    def evaluate_chunk_relevance_to_subquery(self, sub_query: str, chunk_text: str) -> float:
        """Evaluate how relevant a chunk is to a specific sub-query"""
        prompt = f"""
        Rate how relevant this chunk is to answering the specific sub-query.
        
        Sub-query: {sub_query}
        Chunk: {chunk_text[:300]}...
        
        Rate relevance on a scale of 1-5:
        1 = Not relevant at all
        2 = Slightly relevant
        3 = Moderately relevant  
        4 = Highly relevant
        5 = Perfectly relevant
        
        Respond with just the number (1-5):
        """
        
        response = self._call_llm(prompt)
        try:
            return float(response.strip())
        except ValueError:
            return 3.0  # Default moderate relevance
    
    def evaluate_information_synthesis(self, original_query: str, sub_responses: List[str], 
                                    final_response: str) -> Dict[str, float]:
        """Evaluate how well sub-responses were synthesized into final response"""
        prompt = f"""
        Evaluate how well the individual sub-responses were combined into the final response.
        
        Original Query: {original_query}
        
        Sub-responses:
        {chr(10).join([f"{i+1}. {resp}" for i, resp in enumerate(sub_responses)])}
        
        Final Combined Response: {final_response}
        
        Rate on these criteria (1-5 scale):
        1. Information_preservation: How much important information from sub-responses is preserved?
        2. Coherence: Is the final response coherent and well-structured?
        3. Synthesis_quality: How well are the sub-responses integrated (not just concatenated)?
        4. Completeness: Does the final response feel complete for the original query?
        
        Respond in this exact format:
        Information_preservation: X
        Coherence: X
        Synthesis_quality: X
        Completeness: X
        Overall: X
        """
        
        response = self._call_llm(prompt)
        scores = {}
        
        for line in response.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                try:
                    scores[key] = float(value.strip())
                except ValueError:
                    continue
        
        return scores

class RAGASEvaluator:
    """RAGAS metrics implementation for multi-level evaluation"""
    
    def __init__(self):
        self.has_ragas = HAS_RAGAS
        if not self.has_ragas:
            print("Warning: RAGAS not available. Install with: pip install ragas")
    
    def evaluate_with_ragas(self, query: str, answer: str, contexts: List[str], 
                          ground_truth: str = None) -> Dict[str, float]:
        """Evaluate using RAGAS metrics"""
        if not self.has_ragas or not contexts:
            return {}
        
        try:
            # Prepare data for RAGAS
            dataset_dict = {
                'question': [query],
                'answer': [answer],
                'contexts': [contexts],
            }
            
            # Add ground truth if available
            if ground_truth:
                dataset_dict['ground_truth'] = [ground_truth]
            
            dataset = Dataset.from_dict(dataset_dict)
            
            # Define metrics to evaluate
            metrics = [
                answer_relevancy,
                faithfulness,
                context_precision,
                context_relevancy,
            ]
            
            # Add metrics that require ground truth
            if ground_truth:
                metrics.extend([
                    answer_correctness,
                    answer_similarity,
                    context_recall
                ])
            
            # Evaluate
            result = evaluate(dataset, metrics=metrics)
            
            # Convert to dict and handle NaN values
            ragas_scores = {}
            for metric, score in result.items():
                if isinstance(score, (list, np.ndarray)):
                    score = score[0] if len(score) > 0 else 0
                ragas_scores[metric] = float(score) if not np.isnan(float(score)) else 0.0
            
            return ragas_scores
            
        except Exception as e:
            print(f"RAGAS evaluation failed: {e}")
            return {}

class SubQueryEvaluator:
    """Evaluates sub-query identification with hybrid metrics"""
    
    def __init__(self, similarity_threshold: float = 0.7, llm_judge: LLMJudge = None):
        self.similarity_threshold = similarity_threshold
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm_judge = llm_judge
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts"""
        embeddings = self.sentence_model.encode([text1, text2])
        return 1 - cosine(embeddings[0], embeddings[1])
    
    def calculate_coverage_score(self, original_query: str, sub_queries: List[str]) -> float:
        """Calculate how well sub-queries cover the original query"""
        original_embedding = self.sentence_model.encode([original_query])[0]
        
        # Combine sub-queries
        combined_subqueries = " ".join(sub_queries)
        combined_embedding = self.sentence_model.encode([combined_subqueries])[0]
        
        # Calculate coverage as similarity
        coverage = 1 - cosine(original_embedding, combined_embedding)
        return max(0, coverage)
    
    def calculate_redundancy_score(self, sub_queries: List[str]) -> float:
        """Calculate redundancy between sub-queries (lower is better)"""
        if len(sub_queries) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(sub_queries)):
            for j in range(i+1, len(sub_queries)):
                sim = self.compute_semantic_similarity(sub_queries[i], sub_queries[j])
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def find_matches(self, gold_queries: List[str], generated_queries: List[str]) -> Tuple[int, int, int]:
        """Find TP, FP, FN based on semantic similarity"""
        tp = 0
        matched_gold = set()
        matched_generated = set()
        
        # Find true positives
        for i, gen_query in enumerate(generated_queries):
            for j, gold_query in enumerate(gold_queries):
                if j not in matched_gold:
                    similarity = self.compute_semantic_similarity(gen_query, gold_query)
                    if similarity >= self.similarity_threshold:
                        tp += 1
                        matched_gold.add(j)
                        matched_generated.add(i)
                        break
        
        fp = len(generated_queries) - len(matched_generated)
        fn = len(gold_queries) - len(matched_gold)
        
        return tp, fp, fn
    
    def calculate_traditional_metrics(self, tp: int, fp: int, fn: int) -> Dict[str, float]:
        """Calculate precision, recall, F1"""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    def evaluate(self, data: Dict[str, Any]) -> MultiLevelEvaluationResults:
        """Evaluate sub-query identification with hybrid approach"""
        original_query = data.get('query', '')
        gold_queries = data['sub_queries']['gold']
        generated_queries = data['sub_queries']['generated']
        
        # Traditional metrics (matching)
        tp, fp, fn = self.find_matches(gold_queries, generated_queries)
        traditional_metrics = self.calculate_traditional_metrics(tp, fp, fn)
        
        # Calculate average semantic similarity
        if gold_queries and generated_queries:
            similarities = []
            for gen_q in generated_queries:
                max_sim = max([self.compute_semantic_similarity(gen_q, gold_q) for gold_q in gold_queries])
                similarities.append(max_sim)
            traditional_metrics['avg_similarity'] = np.mean(similarities)
        
        # New hybrid metrics
        coverage_score = self.calculate_coverage_score(original_query, generated_queries)
        redundancy_score = self.calculate_redundancy_score(generated_queries)
        
        hybrid_metrics = {
            'coverage_score': coverage_score,
            'redundancy_score': redundancy_score
        }
        
        # LLM-as-a-Judge evaluation
        synthesis_metrics = {}
        if self.llm_judge:
            llm_scores = self.llm_judge.evaluate_subquery_decomposition(original_query, generated_queries)
            synthesis_metrics.update(llm_scores)
        
        return MultiLevelEvaluationResults(
            step_name="Sub-query Identification",
            pooled_metrics=traditional_metrics,
            sub_query_metrics=hybrid_metrics,
            synthesis_metrics=synthesis_metrics,
            additional_info={'similarity_threshold': self.similarity_threshold}
        )

class ChunkRetrievalEvaluator:
    """Evaluates chunk retrieval with multi-level metrics"""
    
    def __init__(self, llm_judge: LLMJudge = None, ragas_evaluator: RAGASEvaluator = None):
        self.llm_judge = llm_judge
        self.ragas_evaluator = ragas_evaluator
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def calculate_precision_recall_at_k(self, retrieved_ids: List[int], relevant_ids: List[int], k: int) -> Tuple[float, float]:
        """Calculate precision@k and recall@k"""
        retrieved_k = retrieved_ids[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant_ids))
        
        precision_k = relevant_retrieved / len(retrieved_k) if retrieved_k else 0
        recall_k = relevant_retrieved / len(relevant_ids) if relevant_ids else 0
        
        return precision_k, recall_k
    
    def evaluate_subquery_retrieval_quality(self, sub_query_data: List[Dict]) -> Dict[str, float]:
        """Evaluate retrieval quality for individual sub-queries"""
        if not sub_query_data or not self.llm_judge:
            return {}
        
        all_relevance_scores = []
        all_precisions = []
        all_recalls = []
        
        for sq_data in sub_query_data:
            sub_query = sq_data.get('sub_query', '')
            retrieved_chunks = sq_data.get('retrieved_chunks', [])
            gold_relevant_ids = sq_data.get('gold_relevant_ids', [])
            
            if not retrieved_chunks:
                continue
            
            # Calculate traditional metrics for this sub-query
            retrieved_ids = [chunk['id'] for chunk in retrieved_chunks]
            if gold_relevant_ids:
                prec_5, rec_5 = self.calculate_precision_recall_at_k(retrieved_ids, gold_relevant_ids, 5)
                all_precisions.append(prec_5)
                all_recalls.append(rec_5)
            
            # Calculate LLM-based relevance for this sub-query
            relevance_scores = []
            for chunk in retrieved_chunks[:5]:  # Limit to top 5 for efficiency
                chunk_text = chunk.get('text', '')
                relevance = self.llm_judge.evaluate_chunk_relevance_to_subquery(sub_query, chunk_text)
                relevance_scores.append(relevance / 5.0)  # Normalize to 0-1
            
            if relevance_scores:
                all_relevance_scores.append(np.mean(relevance_scores))
        
        return {
            'avg_subquery_precision': np.mean(all_precisions) if all_precisions else 0,
            'avg_subquery_recall': np.mean(all_recalls) if all_recalls else 0,
            'avg_relevance_score': np.mean(all_relevance_scores) if all_relevance_scores else 0
        }
    
    def evaluate(self, data: Dict[str, Any]) -> MultiLevelEvaluationResults:
        """Evaluate chunk retrieval with multi-level approach"""
        query = data.get('query', '')
        
        # Pooled evaluation (traditional approach)
        gold_relevant_ids = data['chunks']['gold_relevant_ids']
        all_retrieved_chunks = data['chunks']['retrieved']
        retrieved_ids = [chunk['id'] for chunk in all_retrieved_chunks]
        
        # Calculate pooled traditional metrics
        k_values = [1, 3, 5, 10]
        pooled_metrics = {}
        
        for k in k_values:
            if k <= len(retrieved_ids):
                prec_k, rec_k = self.calculate_precision_recall_at_k(retrieved_ids, gold_relevant_ids, k)
                pooled_metrics[f'precision@{k}'] = prec_k
                pooled_metrics[f'recall@{k}'] = rec_k
        
        # Overall TP, FP, FN for pooled evaluation
        tp = len(set(retrieved_ids) & set(gold_relevant_ids))
        fp = len(set(retrieved_ids) - set(gold_relevant_ids))
        fn = len(set(gold_relevant_ids) - set(retrieved_ids))
        
        pooled_metrics.update({
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'total_retrieved': len(retrieved_ids),
            'total_relevant': len(gold_relevant_ids)
        })
        
        # Sub-query level evaluation
        sub_query_data = data.get('sub_query_retrievals', [])
        sub_query_metrics = self.evaluate_subquery_retrieval_quality(sub_query_data)
        
        # RAGAS evaluation (synthesis metrics)
        synthesis_metrics = {}
        if self.ragas_evaluator:
            contexts = [chunk.get('text', '') for chunk in all_retrieved_chunks]
            # We'll use a placeholder answer for context evaluation
            ragas_scores = self.ragas_evaluator.evaluate_with_ragas(
                query, "placeholder answer", contexts
            )
            synthesis_metrics = {k: v for k, v in ragas_scores.items() if 'context' in k}
        
        # Fallback context precision calculation
        if not synthesis_metrics:
            context_precision = tp / len(retrieved_ids) if retrieved_ids else 0
            synthesis_metrics['context_precision'] = context_precision
        
        return MultiLevelEvaluationResults(
            step_name="Chunk Retrieval",
            pooled_metrics=pooled_metrics,
            sub_query_metrics=sub_query_metrics,
            synthesis_metrics=synthesis_metrics,
            additional_info={}
        )

class ChunkGradingEvaluator:
    """Evaluates chunk grading with multi-level metrics"""
    
    def __init__(self, llm_judge: LLMJudge = None):
        self.llm_judge = llm_judge
    
    def evaluate_subquery_grading(self, sub_query_data: List[Dict]) -> Dict[str, float]:
        """Evaluate grading quality for individual sub-queries"""
        all_accuracies = []
        all_precisions = []
        all_recalls = []
        all_f1s = []
        
        for sq_data in sub_query_data:
            gold_grades = sq_data.get('gold_grades', {})
            predicted_grades = sq_data.get('predicted_grades', {})
            gold_grades = {str(k): v for k, v in gold_grades.items()}
            predicted_grades = {str(k): v for k, v in predicted_grades.items()}
            if not gold_grades or not predicted_grades:
                continue
            
            # Align grades by chunk ID
            common_ids = sorted(set(str(k) for k in gold_grades.keys()) & set(str(k) for k in predicted_grades.keys()))
            if not common_ids:
                continue
            
            gold_values = [gold_grades[id_] for id_ in common_ids]
            pred_values = [predicted_grades[id_] for id_ in common_ids]
            
            # Calculate metrics for this sub-query
            accuracy = accuracy_score(gold_values, pred_values)
            precision, recall, f1, _ = precision_recall_fscore_support(
                gold_values, pred_values, average='binary', zero_division=0
            )
            
            all_accuracies.append(accuracy)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)
        
        return {
            'avg_subquery_accuracy': np.mean(all_accuracies) if all_accuracies else 0,
            'avg_subquery_precision': np.mean(all_precisions) if all_precisions else 0,
            'avg_subquery_recall': np.mean(all_recalls) if all_recalls else 0,
            'avg_subquery_f1': np.mean(all_f1s) if all_f1s else 0
        }
    
    def evaluate(self, data: Dict[str, Any]) -> MultiLevelEvaluationResults:
        """Evaluate chunk grading with multi-level approach"""
        
        # Pooled evaluation (traditional approach)
        gold_grades = data['chunk_grading']['gold_grades']
        predicted_grades = data['chunk_grading']['predicted_grades']

        gold_grades = {str(k): v for k, v in gold_grades.items()}
        predicted_grades = {str(k): v for k, v in predicted_grades.items()}
        print("gold_grades keys:", list(gold_grades.keys()))
        print("predicted_grades keys:", list(predicted_grades.keys()))
        # print("common_ids:", common_ids)
        # Align grades by chunk ID
        common_ids = sorted(set(str(k) for k in gold_grades.keys()) & set(str(k) for k in predicted_grades.keys()))
        if not common_ids:
            return MultiLevelEvaluationResults(
                step_name="Chunk Grading",
                pooled_metrics={},
                sub_query_metrics={},
                synthesis_metrics={},
                additional_info={'error': 'No common chunk IDs found'}
            )
        
        gold_values = [gold_grades[id_] for id_ in common_ids]
        pred_values = [predicted_grades[id_] for id_ in common_ids]
        
        # Pooled traditional metrics
        accuracy = accuracy_score(gold_values, pred_values)
        precision, recall, f1, _ = precision_recall_fscore_support(
            gold_values, pred_values, average='binary', zero_division=0
        )
        
        # MAE for graded relevance
        mae = np.mean(np.abs(np.array(gold_values) - np.array(pred_values)))
        
        # Correlation
        correlation = np.corrcoef(gold_values, pred_values)[0, 1] if len(set(gold_values)) > 1 else 0
        
        pooled_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mae': mae,
            'correlation': correlation if not np.isnan(correlation) else 0
        }
        
        # Sub-query level evaluation
        sub_query_data = data.get('sub_query_gradings', [])
        sub_query_metrics = self.evaluate_subquery_grading(sub_query_data)
        
        return MultiLevelEvaluationResults(
            step_name="Chunk Grading",
            pooled_metrics=pooled_metrics,
            sub_query_metrics=sub_query_metrics,
            synthesis_metrics={},
            additional_info={'num_evaluated_chunks': len(common_ids)}
        )

class AnswerGenerationEvaluator:
    """Evaluates answer generation with multi-level metrics"""
    
    def __init__(self, llm_judge: LLMJudge = None, ragas_evaluator: RAGASEvaluator = None):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm_judge = llm_judge
        self.ragas_evaluator = ragas_evaluator
    
    def calculate_answer_relevance(self, query: str, answer: str) -> float:
        """Calculate answer relevance using semantic similarity"""
        similarity = 1 - cosine(
            self.sentence_model.encode([query])[0],
            self.sentence_model.encode([answer])[0]
        )
        return max(0, similarity)
    
    def calculate_faithfulness_proxy(self, answer: str, context_chunks: List[Dict]) -> float:
        """Calculate faithfulness based on context overlap"""
        if not context_chunks:
            return 0
        
        context_text = " ".join([chunk.get('text', '') for chunk in context_chunks])
        answer_embedding = self.sentence_model.encode([answer])[0]
        context_embedding = self.sentence_model.encode([context_text])[0]
        
        similarity = 1 - cosine(answer_embedding, context_embedding)
        return max(0, similarity)
    
    def evaluate_subresponse_quality(self, sub_query_data: List[Dict]) -> Dict[str, float]:
        """Evaluate quality of individual sub-responses"""
        all_relevance_scores = []
        all_faithfulness_scores = []
        
        for sq_data in sub_query_data:
            sub_query = sq_data.get('sub_query', '')
            sub_response = sq_data.get('sub_response', '')
            context_chunks = sq_data.get('context_chunks', [])
            
            if not sub_response:
                continue
            
            # Calculate relevance of sub-response to sub-query
            relevance = self.calculate_answer_relevance(sub_query, sub_response)
            all_relevance_scores.append(relevance)
            
            # Calculate faithfulness of sub-response to its context
            faithfulness = self.calculate_faithfulness_proxy(sub_response, context_chunks)
            all_faithfulness_scores.append(faithfulness)
        
        return {
            'avg_subresponse_relevance': np.mean(all_relevance_scores) if all_relevance_scores else 0,
            'avg_subresponse_faithfulness': np.mean(all_faithfulness_scores) if all_faithfulness_scores else 0
        }
    
    def evaluate(self, data: Dict[str, Any]) -> MultiLevelEvaluationResults:
        """Evaluate answer generation with multi-level approach"""
        query = data['query']
        generated_answer = data['answer']['generated']
        ground_truth = data['answer'].get('ground_truth', '')
        context_chunks = data['chunks']['retrieved']
        
        # Pooled/Final level evaluation (traditional RAGAS)
        context_text = " ".join([chunk.get('text', '') for chunk in context_chunks])
        contexts = [chunk.get('text', '') for chunk in context_chunks]
        
        # Basic automated metrics
        answer_relevance = self.calculate_answer_relevance(query, generated_answer)
        faithfulness = self.calculate_faithfulness_proxy(generated_answer, context_chunks)
        
        # Answer completeness (similarity with ground truth if available)
        answer_completeness = 0
        if ground_truth:
            answer_completeness = 1 - cosine(
                self.sentence_model.encode([generated_answer])[0],
                self.sentence_model.encode([ground_truth])[0]
            )
            answer_completeness = max(0, answer_completeness)
        
        pooled_metrics = {
            'answer_length': len(generated_answer.split()),
            'query_answer_similarity': answer_relevance,
            'basic_faithfulness': faithfulness,
            'answer_completeness': answer_completeness
        }
        
        # Sub-query level evaluation
        sub_query_data = data.get('sub_query_responses', [])
        sub_query_metrics = self.evaluate_subresponse_quality(sub_query_data)
        
        # Synthesis evaluation using RAGAS and LLM judge
        synthesis_metrics = {}
        
        # RAGAS metrics for final response
        if self.ragas_evaluator and contexts:
            ragas_metrics = self.ragas_evaluator.evaluate_with_ragas(
                query, generated_answer, contexts, ground_truth
            )
            synthesis_metrics.update(ragas_metrics)
        else:
            # Fallback RAGAS-style metrics
            synthesis_metrics = {
                'answer_relevancy': answer_relevance,
                'faithfulness': faithfulness,
                'answer_completeness': answer_completeness
            }
        
        # LLM-as-a-Judge for synthesis quality
        if self.llm_judge and sub_query_data:
            sub_responses = [sq.get('sub_response', '') for sq in sub_query_data]
            if sub_responses:
                llm_synthesis_scores = self.llm_judge.evaluate_information_synthesis(
                    query, sub_responses, generated_answer
                )
                synthesis_metrics.update(llm_synthesis_scores)
        
        return MultiLevelEvaluationResults(
            step_name="Answer Generation",
            pooled_metrics=pooled_metrics,
            sub_query_metrics=sub_query_metrics,
            synthesis_metrics=synthesis_metrics,
            additional_info={}
        )

class VisualizationGenerator:
    """Generate visualizations for multi-level evaluation results"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        plt.style.use('default')
    
    def plot_multilevel_comparison(self, aggregated_results: Dict[str, Any]):
        """Compare pooled vs sub-query level metrics"""
        step_names = []
        pooled_f1 = []
        subquery_f1 = []
        
        for step_name, results in aggregated_results.items():
            if 'pooled_metrics' in results and 'sub_query_metrics' in results:
                pooled = results['pooled_metrics']
                subquery = results['sub_query_metrics']
                
                if 'f1' in pooled and any('f1' in k for k in subquery.keys()):
                    step_names.append(step_name.replace('_', '\n'))
                    pooled_f1.append(pooled['f1']['mean'])
                    
                    # Find sub-query F1 metric
                    sq_f1 = next((v['mean'] for k, v in subquery.items() if 'f1' in k), 0)
                    subquery_f1.append(sq_f1)
        
        if not step_names:
            return None
        
        x = np.arange(len(step_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, pooled_f1, width, label='Pooled/Final Level', alpha=0.8)
        ax.bar(x + width/2, subquery_f1, width, label='Sub-Query Level', alpha=0.8)
        
        ax.set_xlabel('Pipeline Steps')
        ax.set_ylabel('F1 Score')
        ax.set_title('Multi-Level Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(step_names)
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        return fig
    
    def plot_synthesis_metrics(self, synthesis_data: Dict[str, float]):
        """Create radar chart for synthesis quality metrics"""
        if not synthesis_data:
            return None
        
        # Filter synthesis-specific metrics
        synthesis_metrics = {k: v for k, v in synthesis_data.items() 
                           if any(word in k.lower() for word in ['synthesis', 'coherence', 'preservation', 'information'])}
        
        if not synthesis_metrics:
            return None
        
        # Prepare data
        metrics = list(synthesis_metrics.keys())
        values = list(synthesis_metrics.values())
        
        # Normalize values to 0-1 range if needed
        normalized_values = []
        for v in values:
            if v > 1:  # Likely on 1-5 scale
                normalized_values.append(v / 5.0)
            else:
                normalized_values.append(v)
        
        # Number of metrics
        N = len(metrics)
        
        # Angles for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add first value to end to complete the circle
        normalized_values += normalized_values[:1]
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Plot
        ax.plot(angles, normalized_values, 'o-', linewidth=2, label='Synthesis Quality')
        ax.fill(angles, normalized_values, alpha=0.25)
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', '\n') for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Information Synthesis Quality', size=16, y=1.1)
        
        plt.tight_layout()
        return fig
    
    def save_all_plots(self, results: Dict[str, Any], output_dir: str = "evaluation_plots"):
        """Save all visualization plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot multi-level comparison
        multilevel_fig = self.plot_multilevel_comparison(results)
        if multilevel_fig:
            multilevel_fig.savefig(f"{output_dir}/multilevel_comparison.png", dpi=300, bbox_inches='tight')
            plt.close(multilevel_fig)
        
        # Plot synthesis metrics for answer generation
        if 'answer_generation' in results and 'synthesis_metrics' in results['answer_generation']:
            synthesis_metrics = {}
            for k, v in results['answer_generation']['synthesis_metrics'].items():
                if isinstance(v, dict) and 'mean' in v:
                    synthesis_metrics[k] = v['mean']
            
            synthesis_fig = self.plot_synthesis_metrics(synthesis_metrics)
            if synthesis_fig:
                synthesis_fig.savefig(f"{output_dir}/synthesis_quality.png", dpi=300, bbox_inches='tight')
                plt.close(synthesis_fig)

class HybridRAGPipelineEvaluator:
    """Main evaluator for the hybrid multi-level RAG pipeline"""
    
    def __init__(self, gemini_api_key: str = None, llm_model: str = "gpt-4"):
        # Initialize LLM Judge
        self.llm_judge = LLMJudge(api_key=gemini_api_key, model=llm_model)
        
        # Initialize RAGAS evaluator
        self.ragas_evaluator = RAGASEvaluator()
        
        # Initialize step evaluators
        self.sub_query_evaluator = SubQueryEvaluator(llm_judge=self.llm_judge)
        self.retrieval_evaluator = ChunkRetrievalEvaluator(
            llm_judge=self.llm_judge, 
            ragas_evaluator=self.ragas_evaluator
        )
        self.grading_evaluator = ChunkGradingEvaluator(llm_judge=self.llm_judge)
        self.generation_evaluator = AnswerGenerationEvaluator(
            llm_judge=self.llm_judge,
            ragas_evaluator=self.ragas_evaluator
        )
        
        # Initialize visualizer
        self.visualizer = VisualizationGenerator()
    
    def evaluate_single_sample(self, data: Dict[str, Any]) -> Dict[str, MultiLevelEvaluationResults]:
        """Evaluate a single sample through all pipeline steps"""
        results = {}
        
        # Evaluate each step
        if 'sub_queries' in data:
            results['sub_query_identification'] = self.sub_query_evaluator.evaluate(data)
        
        if 'chunks' in data:
            results['chunk_retrieval'] = self.retrieval_evaluator.evaluate(data)
        
        if 'chunk_grading' in data:
            results['chunk_grading'] = self.grading_evaluator.evaluate(data)
        
        if 'answer' in data:
            results['answer_generation'] = self.generation_evaluator.evaluate(data)
        
        return results
    
    def evaluate_dataset(self, dataset: List[Dict[str, Any]], save_plots: bool = True, 
                        output_dir: str = "evaluation_output") -> Dict[str, Any]:
        """Evaluate entire dataset and aggregate results"""
        print(f"Evaluating {len(dataset)} samples with hybrid multi-level approach...")
        
        all_results = []
        sample_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for i, sample in enumerate(dataset):
            if i % 10 == 0:
                print(f"Processing sample {i+1}/{len(dataset)}")
            
            sample_results = self.evaluate_single_sample(sample)
            all_results.append(sample_results)
            
            # Collect metrics for distribution analysis
            for step_name, result in sample_results.items():
                for metric_name, value in result.pooled_metrics.items():
                    sample_metrics[step_name]['pooled'][metric_name].append(value)
                for metric_name, value in result.sub_query_metrics.items():
                    sample_metrics[step_name]['sub_query'][metric_name].append(value)
                for metric_name, value in result.synthesis_metrics.items():
                    sample_metrics[step_name]['synthesis'][metric_name].append(value)
        
        # Aggregate results
        aggregated = self._aggregate_results(all_results)
        
        # Add sample metrics for visualization
        for step_name in aggregated:
            if step_name in sample_metrics:
                aggregated[step_name]['sample_metrics'] = dict(sample_metrics[step_name])
        
        # Generate and save visualizations
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            self.visualizer.save_all_plots(aggregated, f"{output_dir}/plots")
        
        return aggregated
    
    def _aggregate_results(self, all_results: List[Dict[str, MultiLevelEvaluationResults]]) -> Dict[str, Any]:
        """Aggregate results across all samples"""
        aggregated = {}
        
        # Get all step names
        step_names = set()
        for results in all_results:
            step_names.update(results.keys())
        
        for step_name in step_names:
            step_results = [results[step_name] for results in all_results if step_name in results]
            
            if not step_results:
                continue
            
            # Aggregate pooled metrics
            pooled_metrics = {}
            for metric_name in step_results[0].pooled_metrics.keys():
                values = [r.pooled_metrics[metric_name] for r in step_results 
                         if metric_name in r.pooled_metrics and not np.isnan(r.pooled_metrics[metric_name])]
                if values:
                    pooled_metrics[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values)
                    }
            
            # Aggregate sub-query metrics
            sub_query_metrics = {}
            if step_results[0].sub_query_metrics:
                for metric_name in step_results[0].sub_query_metrics.keys():
                    values = [r.sub_query_metrics[metric_name] for r in step_results 
                             if metric_name in r.sub_query_metrics and not np.isnan(r.sub_query_metrics[metric_name])]
                    if values:
                        sub_query_metrics[metric_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values),
                            'median': np.median(values)
                        }
            
            # Aggregate synthesis metrics
            synthesis_metrics = {}
            if step_results[0].synthesis_metrics:
                for metric_name in step_results[0].synthesis_metrics.keys():
                    values = [r.synthesis_metrics[metric_name] for r in step_results 
                             if metric_name in r.synthesis_metrics and not np.isnan(r.synthesis_metrics[metric_name])]
                    if values:
                        synthesis_metrics[metric_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values),
                            'median': np.median(values)
                        }
            
            aggregated[step_name] = {
                'pooled_metrics': pooled_metrics,
                'sub_query_metrics': sub_query_metrics,
                'synthesis_metrics': synthesis_metrics,
                'sample_count': len(step_results)
            }
        
        return aggregated
    
    def generate_hybrid_report(self, results: Dict[str, Any], output_file: str = None) -> str:
        """Generate a comprehensive hybrid evaluation report"""
        report = "HYBRID RAG PIPELINE EVALUATION REPORT\n"
        report += "=" * 70 + "\n\n"
        
        # Executive Summary
        report += "EXECUTIVE SUMMARY\n"
        report += "-" * 17 + "\n"
        
        overall_scores = []
        synthesis_scores = []
        
        for step_name, step_results in results.items():
            # Get F1 scores from pooled metrics
            if 'pooled_metrics' in step_results and 'f1' in step_results['pooled_metrics']:
                f1_score = step_results['pooled_metrics']['f1']['mean']
                overall_scores.append(f1_score)
                report += f"• {step_name.replace('_', ' ').title()} (Pooled): F1 = {f1_score:.3f}\n"
            
            # Get synthesis quality scores
            if 'synthesis_metrics' in step_results and step_results['synthesis_metrics']:
                synthesis_avg = np.mean([v['mean'] for v in step_results['synthesis_metrics'].values() 
                                       if isinstance(v, dict) and 'mean' in v])
                if not np.isnan(synthesis_avg):
                    synthesis_scores.append(synthesis_avg)
                    report += f"• {step_name.replace('_', ' ').title()} (Synthesis): Quality = {synthesis_avg:.3f}\n"
        
        if overall_scores:
            report += f"• Overall Pipeline Performance: {np.mean(overall_scores):.3f}\n"
        if synthesis_scores:
            report += f"• Overall Synthesis Quality: {np.mean(synthesis_scores):.3f}\n"
        report += "\n"
        
        # Detailed Results by Step
        for step_name, step_results in results.items():
            report += f"{step_name.upper().replace('_', ' ')}\n"
            report += "=" * len(step_name) + "\n"
            
            # Pooled/Final Level Metrics
            if 'pooled_metrics' in step_results and step_results['pooled_metrics']:
                report += "📊 Pooled/Final Level Metrics:\n"
                for metric, stats in step_results['pooled_metrics'].items():
                    report += f"   {metric.replace('_', ' ').title():<25}: "
                    report += f"{stats['mean']:.4f} ± {stats['std']:.4f} "
                    report += f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})\n"
                report += "\n"
            
            # Sub-Query Level Metrics
            if 'sub_query_metrics' in step_results and step_results['sub_query_metrics']:
                report += "🔍 Sub-Query Level Metrics:\n"
                for metric, stats in step_results['sub_query_metrics'].items():
                    report += f"   {metric.replace('_', ' ').title():<25}: "
                    report += f"{stats['mean']:.4f} ± {stats['std']:.4f}\n"
                report += "\n"
            
            # Synthesis/Quality Metrics
            if 'synthesis_metrics' in step_results and step_results['synthesis_metrics']:
                report += "🔀 Synthesis/Quality Metrics:\n"
                for metric, stats in step_results['synthesis_metrics'].items():
                    report += f"   {metric.replace('_', ' ').title():<25}: "
                    report += f"{stats['mean']:.4f} ± {stats['std']:.4f}\n"
                report += "\n"
            
            report += f"Sample Count: {step_results['sample_count']}\n"
            report += "\n" + "="*70 + "\n\n"
        
        # Recommendations
        report += "RECOMMENDATIONS\n"
        report += "-" * 15 + "\n"
        
        recommendations_added = False
        
        for step_name, step_results in results.items():
            # Check pooled metrics
            if 'pooled_metrics' in step_results:
                metrics = step_results['pooled_metrics']
                if 'f1' in metrics and metrics['f1']['mean'] < 0.7:
                    report += f"⚠️  {step_name.replace('_', ' ').title()} (Pooled): F1 score ({metrics['f1']['mean']:.3f}) below recommended threshold (0.7)\n"
                    recommendations_added = True
                if 'precision' in metrics and 'recall' in metrics:
                    prec = metrics['precision']['mean']
                    rec = metrics['recall']['mean']
                    if prec > rec + 0.1:
                        report += f"📈 {step_name.replace('_', ' ').title()}: Consider improving recall (precision: {prec:.3f}, recall: {rec:.3f})\n"
                        recommendations_added = True
                    elif rec > prec + 0.1:
                        report += f"📉 {step_name.replace('_', ' ').title()}: Consider improving precision (precision: {prec:.3f}, recall: {rec:.3f})\n"
                        recommendations_added = True
            
            # Check synthesis quality
            if 'synthesis_metrics' in step_results and step_results['synthesis_metrics']:
                for metric, stats in step_results['synthesis_metrics'].items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        if 'synthesis' in metric.lower() and stats['mean'] < 3.5:  # Assuming 1-5 scale
                            report += f"🔀 {step_name.replace('_', ' ').title()}: {metric.replace('_', ' ').title()} score ({stats['mean']:.3f}) could be improved\n"
                            recommendations_added = True
        
        if not recommendations_added:
            report += "✅ All metrics are within acceptable ranges!\n"
        
        report += "\n"
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Hybrid evaluation report saved to {output_file}")
        
        return report

# Enhanced sample data structure for hybrid evaluation
def create_hybrid_sample_data():
    """Create sample data for hybrid multi-level evaluation"""
    return [
        {
            "query": "What are the benefits of renewable energy?",
            "sub_queries": {
                "gold": ["What are environmental benefits of renewable energy?", 
                        "What are economic benefits of renewable energy?"],
                "generated": ["What are environmental benefits of renewable energy?"]
            },
            "chunks": {
                "gold_relevant_ids": [1, 3, 5],
                "retrieved": [
                    {"id": 1, "text": "Renewable energy sources like solar and wind significantly reduce carbon emissions compared to fossil fuels, helping combat climate change.", "score": 0.9},
                    {"id": 2, "text": "Solar panels require significant upfront investment but provide long-term savings through reduced electricity bills.", "score": 0.8},
                    {"id": 3, "text": "The renewable energy sector has created millions of jobs worldwide, from manufacturing to installation and maintenance.", "score": 0.7},
                    {"id": 4, "text": "Coal power plants are major sources of air pollution and greenhouse gas emissions.", "score": 0.6},
                    {"id": 5, "text": "Wind energy costs have decreased by 70% in the last decade, making it cost-competitive with traditional energy sources.", "score": 0.5}
                ]
            },
            "chunk_grading": {
                "gold_grades": {1: 1, 2: 1, 3: 1, 4: 0, 5: 1},
                "predicted_grades": {1: 1, 2: 0, 3: 1, 4: 1, 5: 0}
            },
            "answer": {
                "generated": "Renewable energy has significant environmental and economic benefits. Environmentally, it reduces carbon emissions and helps combat climate change. Economically, it creates jobs and provides long-term cost savings despite initial investment costs.",
                "ground_truth": "Renewable energy provides clean power by reducing carbon emissions, creates employment opportunities in manufacturing and installation, and offers long-term economic benefits through decreased operational costs."
            },
            # New hybrid-specific data
            "sub_query_retrievals": [
                {
                    "sub_query": "Environmental impact of renewable energy",
                    "retrieved_chunks": [
                        {"id": 1, "text": "Renewable energy sources like solar and wind significantly reduce carbon emissions compared to fossil fuels, helping combat climate change.", "score": 0.9},
                        {"id": 4, "text": "Coal power plants are major sources of air pollution and greenhouse gas emissions.", "score": 0.6}
                    ],
                    "gold_relevant_ids": [1, 4]
                },
                {
                    "sub_query": "Cost effectiveness of renewable energy",
                    "retrieved_chunks": [
                        {"id": 2, "text": "Solar panels require significant upfront investment but provide long-term savings through reduced electricity bills.", "score": 0.8},
                        {"id": 5, "text": "Wind energy costs have decreased by 70% in the last decade, making it cost-competitive with traditional energy sources.", "score": 0.5}
                    ],
                    "gold_relevant_ids": [2, 5]
                }
            ],
            "sub_query_gradings": [
                {
                    "sub_query": "Environmental impact of renewable energy",
                    "gold_grades": {1: 1, 4: 1},
                    "predicted_grades": {1: 1, 4: 0}
                },
                {
                    "sub_query": "Cost effectiveness of renewable energy",
                    "gold_grades": {2: 1, 5: 1},
                    "predicted_grades": {2: 0, 5: 1}
                }
            ],
            "sub_query_responses": [
                {
                    "sub_query": "Environmental impact of renewable energy",
                    "sub_response": "Renewable energy significantly reduces carbon emissions and helps combat climate change compared to fossil fuels.",
                    "context_chunks": [
                        {"id": 1, "text": "Renewable energy sources like solar and wind significantly reduce carbon emissions compared to fossil fuels, helping combat climate change."}
                    ]
                },
                {
                    "sub_query": "Cost effectiveness of renewable energy",
                    "sub_response": "Renewable energy provides long-term cost savings and has become increasingly cost-competitive.",
                    "context_chunks": [
                        {"id": 5, "text": "Wind energy costs have decreased by 70% in the last decade, making it cost-competitive with traditional energy sources."}
                    ]
                }
            ]
        }
    ]

if __name__ == "__main__":
    # Example usage with hybrid multi-level evaluation
    
    # Initialize evaluator (add your OpenAI API key for LLM-as-a-Judge)
    evaluator = HybridRAGPipelineEvaluator(
        gemini_api_key="",  # Add your API key here or set OPENAI_API_KEY env var
        llm_model="gemini-2.0-flash"
    )
    
    # Load your data (replace with actual data loading)
    sample_data = create_hybrid_sample_data()
    
    print("Starting Hybrid RAG Pipeline Evaluation...")
    print(f"Evaluating {len(sample_data)} samples with multi-level hybrid approach\n")
    
    # Evaluate dataset with hybrid pipeline
    results = evaluator.evaluate_dataset(
        sample_data, 
        save_plots=True, 
        output_dir="hybrid_evaluation_output"
    )
    
    # Generate comprehensive hybrid report
    report = evaluator.generate_hybrid_report(
        results, 
        "hybrid_evaluation_output/hybrid_report.txt"
    )
    print(report)
    
    # Save results as JSON for further analysis
    with open("hybrid_evaluation_output/hybrid_detailed_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✅ Hybrid evaluation complete!")
    print("📁 Output files:")
    print("   • hybrid_evaluation_output/hybrid_report.txt")
    print("   • hybrid_evaluation_output/hybrid_detailed_results.json")
    print("   • hybrid_evaluation_output/plots/ (visualization files)")