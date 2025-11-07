"""
Answer Extraction and Evaluation Module
Handles scoring and accuracy calculation for multiple-choice question answering models.
Supports various answer extraction methods including regex patterns and similarity matching.
"""

import re
import json
import os
import difflib
from collections import defaultdict
from typing import List, Dict, Tuple, Any


class AnswerMatcher:
    """Handles answer extraction and matching operations."""
    
    @staticmethod
    def calculate_string_similarity(first_string: str, second_string: str) -> float:
        """
        Compute similarity ratio between two strings using sequence matching.
        
        Args:
            first_string: First string to compare
            second_string: Second string to compare
            
        Returns:
            Similarity ratio between 0 and 1
        """
        sequence_matcher = difflib.SequenceMatcher(None, first_string, second_string)
        return sequence_matcher.ratio()
    
    @staticmethod
    def find_most_similar_index(string_list: List[str], target_string: str) -> int:
        """
        Given a list of strings and a target string, returns the index of the most similar string in the list.
        
        Args:
            string_list: List of candidate strings
            target_string: Target string to match against
            
        Returns:
            Index of the most similar string
        """
        # Initialize variables to keep track of the most similar string and its index
        most_similar_str = None
        most_similar_index = None
        highest_similarity = 0
        
        # Iterate through each string in the list
        for i, str in enumerate(string_list):
            # Calculate the similarity between the current string and the target string
            similarity = AnswerMatcher.calculate_string_similarity(str, target_string)
            
            # If the current string is more similar than the previous most similar string, update the variables
            if similarity >= highest_similarity:
                most_similar_str = str
                most_similar_index = i
                highest_similarity = similarity

        return most_similar_index
    
    @staticmethod
    def extract_answer_from_text(text: str, options: Dict[str, str]) -> Tuple[List[str], int]:
        """
        Extract answer choice from model output text using multiple strategies.
        
        Attempts extraction in order:
        1. Strict pattern matching for "answer is X" format
        2. General pattern matching for isolated option letters
        3. Option text substring matching
        4. Fuzzy similarity matching as fallback
        
        Args:
            text: Model output text to parse
            options: Dictionary mapping option letters to option text
            
        Returns:
            Tuple of ([first_answer, last_answer], extraction_method_type)
        """
        # Extract final response from reasoning output
        if '## Final Response\n\n' in text:
            text = text.split('## Final Response\n\n')[-1]
        
        # Strategy 1: for strict prompt format
        strict_pattern_matches = list(re.finditer(r"(answer is\s*?)([A-N])", text, re.S))
        if strict_pattern_matches:
            first_answer = strict_pattern_matches[0].group(2)
            last_answer = strict_pattern_matches[-1].group(2)
            return [first_answer, last_answer], 1

        # Strategy 2: non strict pattern matching
        valid_option_letters = 'ABCDEFGHIJKLMN'[:len(options)]
        general_pattern_matches = list(re.finditer(
            r"(is |\*|\W|\ |\(|^|'|\"|#)(?![aA] )([" + valid_option_letters + r"])(\W|$)", 
            text, 
            re.S
        ))
        if general_pattern_matches:
            first_answer = general_pattern_matches[0].group(2)
            last_answer = general_pattern_matches[-1].group(2)
            return [first_answer, last_answer], 1

        # Strategy 3: substring matching in lowercase
        text_lowercase = text.lower()
        option_positions = [
            (opt_key, text_lowercase.rindex(options[opt_key].lower())) 
            for opt_key in options 
            if options[opt_key].lower() in text_lowercase
        ]
        
        if len(option_positions) > 0:
            last_answer = sorted(option_positions, key=lambda x: x[1], reverse=True)[0][0]
            option_positions = [
                (opt_key, text_lowercase.index(options[opt_key].lower())) 
                for opt_key in options 
                if options[opt_key].lower() in text_lowercase
            ]
            first_answer = sorted(option_positions, key=lambda x: x[1], reverse=True)[0][0]
            return [first_answer, last_answer], 2
        else:
            # Strategy 4: fuzzy matching as fallback
            option_labels = [x for x in options]
            option_answers = [options[x].lower() for x in options]
            best_match_index = AnswerMatcher.find_most_similar_index(
                option_answers, 
                text_lowercase
            )
            return [option_labels[best_match_index], option_labels[best_match_index]], 3
    
    @staticmethod
    def verify_ground_truth_match(prediction: str, ground_truth: List[str]) -> int:
        """
        Check if prediction matches any ground truth answer.
        
        Args:
            prediction: Predicted answer text
            ground_truth: List of acceptable ground truth answers
            
        Returns:
            1 if match found, 0 otherwise
        """
        for gt in ground_truth:
            match_result = re.search(
                r"(\W|^)(" + re.escape(gt) + r")(\W|$)", 
                prediction.lower(), 
                re.S
            )
            if match_result:
                return 1
        return 0


class EvaluationScorer:
    """Calculates accuracy scores and manages evaluation metrics."""
    
    @staticmethod
    def compute_accuracy_scores(
        data: List[Dict[str, Any]], 
        ignore_miss: bool = False
    ) -> Tuple[Dict[str, List], List[Dict], List[Dict]]:
        """
        Score model predictions against ground truth answers.
        
        Calculates head match (first extracted answer) and tail match (last extracted answer)
        accuracy scores, grouped by data source.
        
        Args:
            data: List of evaluation items with predictions and ground truth
            ignore_miss: If True, skip items where answer extraction failed
            
        Returns:
            Tuple of (scores_by_source, incorrect_items, correct_items)
        """
        scores_by_source = {}
        incorrect_items = []
        correct_items = []
        
        for data_item in data:
            # Ensure source field exists
            if 'source' not in data_item:
                data_item['source'] = 'unknown'
            
            # Initialize source tracking if needed
            if data_item['source'] not in scores_by_source:
                scores_by_source[data_item['source']] = [0, 0, 0, 0]

            model_output = data_item['output']
            extracted_answers, extraction_type = AnswerMatcher.extract_answer_from_text(
                model_output, 
                data_item['options']
            )
            
            # Skip if extraction failed and ignore_miss is enabled
            if ignore_miss and extraction_type != 1:
                continue

            # Store extracted answers in data item
            data_item['ans'] = extracted_answers
            data_item['ans_type'] = extraction_type

            # Check first answer (head match)
            if extracted_answers[0].lower() == data_item['answer_idx'].lower():
                scores_by_source[data_item['source']][1] += 1
                correct_items.append(data_item)
            else:
                incorrect_items.append(data_item)
            
            # Check last answer (tail match)
            if extracted_answers[1].lower() == data_item['answer_idx'].lower():
                scores_by_source[data_item['source']][3] += 1

            # Increment total count
            scores_by_source[data_item['source']][2] += 1

        # Calculate final scores - use best of head or tail match
        for source_key in scores_by_source:
            head_accuracy = scores_by_source[source_key][1] / scores_by_source[source_key][2]
            tail_accuracy = scores_by_source[source_key][3] / scores_by_source[source_key][2]
            
            if head_accuracy > tail_accuracy:
                scores_by_source[source_key][0] = head_accuracy
            else:
                scores_by_source[source_key][0] = tail_accuracy

        return scores_by_source, incorrect_items, correct_items


def get_results(res_path: str) -> None:
    """
    Load results, compute accuracy metrics, and save to file.
    
    Args:
        res_path: Path to results JSON file
    """
    with open(res_path) as f:
        data = json.load(f) 

    scores, incorrect_items, correct_items = EvaluationScorer.compute_accuracy_scores(data)

    print(f"*{os.path.basename(res_path)}*")
    print(json.dumps(scores, indent=4))
    
    # save results
    with open('result_' + os.path.basename(res_path), 'w') as fw:
        json.dump(scores, fw, ensure_ascii=False, indent=2)


# Legacy function aliases for backward compatibility
def str_similarity(str1, str2):
    """Legacy wrapper for string similarity calculation."""
    return AnswerMatcher.calculate_string_similarity(str1, str2)

def find_most_similar_index(str_list, target_str):
    """Legacy wrapper for finding most similar string index."""
    return AnswerMatcher.find_most_similar_index(str_list, target_str)

def match_choice(text, options):
    """Legacy wrapper for answer extraction."""
    return AnswerMatcher.extract_answer_from_text(text, options)

def match(prediction, ground_truth):
    """Legacy wrapper for ground truth matching."""
    return AnswerMatcher.verify_ground_truth_match(prediction, ground_truth)

def score(data, ignore_miss=False):
    """Legacy wrapper for accuracy scoring."""
    return EvaluationScorer.compute_accuracy_scores(data, ignore_miss)


# if __name__ == "__main__":
#     get_results('output_file_path')