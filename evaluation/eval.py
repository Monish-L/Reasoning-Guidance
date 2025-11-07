"""
Model Evaluation Pipeline
A comprehensive script for evaluating language models on multiple-choice questions
using local API endpoints with batch processing capabilities.
"""

import argparse
import json
import os
from typing import List, Dict, Any, Tuple

from tqdm import tqdm
import openai
from jinja2 import Template
from transformers import AutoTokenizer

from accuracy import get_results


class ModelEvaluator:
    """Handles model evaluation and inference operations."""
    
    def __init__(self, config: argparse.Namespace):
        """
        Initialize the evaluator with configuration parameters.
        
        Args:
            config: Parsed command-line arguments containing model and evaluation settings
        """
        self.config = config
        self.tokenizer = None
        self.chat_template = None
        self.api_client = None
        
        self._initialize_api_client()
        if self.config.use_chat_template:
            self._initialize_tokenizer()
    
    def _initialize_api_client(self) -> None:
        """Set up OpenAI API client with local server configuration."""
        print(f"Using local API server at port {self.config.port}")
        self.api_client = openai.Client(
            base_url=f"http://127.0.0.1:{self.config.port}/v1", 
            api_key="EMPTY"
        )
    
    def _initialize_tokenizer(self) -> None:
        """Initialize tokenizer and extract chat template for prompt formatting."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, 
            trust_remote_code=True, 
            padding_side='left'
        )
        self.chat_template = Template(self.tokenizer.chat_template)
    
    def clean_model_output(self, raw_prediction: str) -> str:
        """
        Post-process raw model output by removing special tokens and whitespace.
        
        Args:
            raw_prediction: Raw text output from the model
            
        Returns:
            Cleaned prediction string
        """
        cleaned = raw_prediction.replace("</s>", "")
        if len(cleaned) > 0 and cleaned[0] == " ":
            cleaned = cleaned[1:]
        return cleaned
    
    def apply_token_limit(self, prompt_list: List[str]) -> List[str]:
        """
        Truncate prompts to maximum token length if specified.
        
        Args:
            prompt_list: List of prompt strings to potentially truncate
            
        Returns:
            List of prompts, truncated if necessary
        """
        if self.config.max_tokens <= 0:
            return prompt_list
        
        truncated_prompts = []
        for single_prompt in prompt_list:
            token_ids = self.tokenizer.encode(single_prompt, add_special_tokens=False)
            
            if len(token_ids) > self.config.max_tokens:
                # Truncate to max tokens and decode back
                token_ids = token_ids[:self.config.max_tokens]
                truncated_prompts.append(self.tokenizer.decode(token_ids))
            else:
                # Use character-based truncation as fallback
                truncated_prompts.append(single_prompt[-self.config.max_tokens:])
        
        return truncated_prompts
    
    def format_with_chat_template(self, prompt_list: List[str]) -> List[str]:
        """
        Apply chat template formatting to prompts if enabled.
        
        Args:
            prompt_list: List of raw prompt strings
            
        Returns:
            List of formatted prompts
        """
        if not self.config.use_chat_template:
            return prompt_list
        
        formatted = [
            self.chat_template.render(
                messages=[{"role": "user", "content": prom}],
                bos_token=self.tokenizer.bos_token,
                add_generation_prompt=True
            ) 
            for prom in prompt_list
        ]
        return formatted
    
    def generate_predictions(
        self, 
        prompt_batch: List[str], 
        model_ref: Any, 
        max_new_tokens: int = 50, 
        print_example: bool = False
    ) -> Tuple[List[str], List[str]]:
        """
        Generate model predictions for a batch of prompts.
        
        Args:
            prompt_batch: List of input prompts
            model_ref: Model reference (unused in API mode)
            max_new_tokens: Maximum tokens to generate per prompt
            print_example: Whether to print first prompt as example
            
        Returns:
            Tuple of (processed predictions, raw predictions)
        """
        sampling_temperature = 0.5
        
        if print_example:
            print("Example:")
            print(prompt_batch[1])
        
        # Apply chat template formatting
        formatted_prompts = self.format_with_chat_template(prompt_batch)
        
        # Apply token limits if configured
        limited_prompts = self.apply_token_limit(formatted_prompts)
        
        # Call API for batch inference
        api_response = self.api_client.completions.create(
            model="default",
            prompt=limited_prompts,
            temperature=sampling_temperature, 
            top_p=0.9, 
            max_tokens=max_new_tokens
        )
        
        # Extract predictions from response
        raw_predictions = [choice.text for choice in api_response.choices]
        
        # Clean up predictions
        processed_predictions = [
            self.clean_model_output(pred) for pred in raw_predictions
        ]
        
        return processed_predictions, raw_predictions


def load_evaluation_data(filepath: str) -> List[Dict[str, Any]]:
    """
    Load evaluation dataset from JSON file.
    
    Handles both list format and dictionary with source keys.
    
    Args:
        filepath: Path to JSON file containing evaluation data
        
    Returns:
        List of evaluation items with source metadata
    """
    with open(filepath, 'r') as file_handle:
        raw_data = json.load(file_handle)
    
    # Normalize data structure
    evaluation_items = []
    if isinstance(raw_data, list):
        raw_data = {'normal': raw_data}
    
    # Flatten and add source metadata
    for source_key, items in raw_data.items():
        for data_item in items:
            data_item['source'] = source_key
        evaluation_items.extend(items)
    
    return evaluation_items


def build_question_prompt(item: Dict[str, Any], strict_mode: bool) -> str:
    """
    Construct the prompt string for a multiple-choice question.
    
    Args:
        item: Question item with 'question' and 'options' fields
        strict_mode: Whether to use strict prompt format
        
    Returns:
        Formatted prompt string
    """
    # Format options as string
    options_text = '\n'.join([
        f'{option_key}. {option_value}' 
        for option_key, option_value in item['options'].items()
    ])
    item['option_str'] = options_text
    
    # Select prompt template based on mode
    if strict_mode:
        template = "Please answer the following multiple-choice questions. Please answer the following multiple-choice questions, ensuring your response concludes with the correct option in the format: 'The answer is A.'.\n{question}\n{option_str}"
    else:
        template = "Please answer the following multiple-choice question:\n{question}\n{option_str}"
    
    # Format and return prompt
    return template.format_map(item)


def construct_output_filename(config: argparse.Namespace) -> str:
    """
    Generate output filename based on configuration parameters.
    
    Args:
        config: Configuration namespace with model and eval settings
        
    Returns:
        Formatted output filename
    """
    model_identifier = os.path.split(config.model_name)[-1]
    eval_basename = os.path.basename(config.eval_file).replace('.json', '')
    
    filename_parts = [
        model_identifier,
        eval_basename,
        f'_{config.task}',
        '_strict-prompt' if config.strict_prompt else ''
    ]
    
    return ''.join(filename_parts) + '.json'


def execute_evaluation_pipeline():
    """Main execution function for the evaluation pipeline."""
    
    # Parse command-line arguments
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        '--model_name', 
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf"
    )
    argument_parser.add_argument('--eval_file', type=str, required=True)
    argument_parser.add_argument('--max_new_tokens', type=int, default=2000)
    argument_parser.add_argument('--max_tokens', type=int, default=-1)
    argument_parser.add_argument('--use_chat_template', type=bool, default=True)
    argument_parser.add_argument('--strict_prompt', action="store_true")
    argument_parser.add_argument('--task', type=str, default='api')
    argument_parser.add_argument('--port', type=int, default=30000)
    argument_parser.add_argument('--batch_size', type=int, default=1024)
    
    configuration = argument_parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(configuration)
    
    # Load evaluation dataset
    evaluation_dataset = load_evaluation_data(configuration.eval_file)
    
    # Storage for results
    collected_results = []
    model_reference = None
    
    # Process dataset in batches
    total_batches = len(evaluation_dataset) // configuration.batch_size + 1
    
    for batch_index in tqdm(range(total_batches)):
        # Extract current batch
        batch_start = batch_index * configuration.batch_size
        batch_end = (batch_index + 1) * configuration.batch_size
        current_batch = evaluation_dataset[batch_start:batch_end]
        
        if len(current_batch) == 0:
            break
        
        # Prepare prompts for batch
        batch_prompts = [
            build_question_prompt(item, configuration.strict_prompt)
            for item in current_batch
        ]
        
        # Store formatted prompts in items
        for item, formatted_prompt in zip(current_batch, batch_prompts):
            item["input_str"] = formatted_prompt
        
        # Extract just the prompt strings
        prompt_strings = [item["input_str"] for item in current_batch]
        
        # Generate predictions
        show_example = (batch_index == 0)
        predictions, _ = evaluator.generate_predictions(
            prompt_strings, 
            model_ref=model_reference, 
            max_new_tokens=configuration.max_new_tokens, 
            print_example=show_example
        )
        
        # Collect results with predictions
        for item_index, item in enumerate(current_batch):
            prediction = predictions[item_index]
            if len(prediction) == 0:
                continue
            item["output"] = prediction
            collected_results.append(item)
    
    # Generate output filename
    output_filename = construct_output_filename(configuration)
    
    # Save results to file
    with open(output_filename, 'w') as output_file:
        json.dump(
            collected_results, 
            output_file, 
            ensure_ascii=False, 
            indent=2
        )
    
    # Compute and display accuracy metrics
    get_results(output_filename)


if __name__ == "__main__":
    execute_evaluation_pipeline()