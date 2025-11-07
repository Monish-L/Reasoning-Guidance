"""
Supervised Fine-Tuning (SFT) Pipeline for Large Language Models
Distributed training script using Accelerate, DeepSpeed, and WandB for experiment tracking.
Supports gradient checkpointing, mixed precision training, and distributed data parallel.
"""

import os
import json
import logging
import argparse
import shutil
import traceback
from typing import Dict, List, Any

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
from jinja2 import Template
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    set_seed, 
    get_cosine_schedule_with_warmup
)

# Set file creation permissions
os.umask(0)

# HuggingFace authentication token
os.environ["HF_TOKEN"] = ""

# Initialize logging
logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')


class MedicalReasoningDataset(torch.utils.data.Dataset):
    """
    Dataset class for medical reasoning fine-tuning.
    Processes question-answer pairs with guided solutions into training examples.
    """
    
    def __init__(self, config: argparse.Namespace, tokenizer: AutoTokenizer):
        """
        Initialize dataset with configuration and tokenizer.
        
        Args:
            config: Training configuration containing data path and sequence length
            tokenizer: Pre-trained tokenizer for encoding text
        """
        self.config = config
        self.tokenizer = tokenizer
        
        # Load training data from JSON file
        with open(config.data_path) as file_handle:
            self.data = json.load(file_handle)
        
        # Filter and prepare data
        filtered_data = []
        for data_item in self.data:
            filtered_data.append(data_item)
        print('Filtered out', len(self.data), len(filtered_data))
        self.data = filtered_data

        self.max_seq_len = self.config.max_seq_len
        self.debug = 0

        # Using Llama3-instruct chat template for base model training
        llama3_chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        
        # Set chat template if not already present
        if not tokenizer.chat_template:
            tokenizer.chat_template = llama3_chat_template
            
        self.template = Template(tokenizer.chat_template)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Retrieve a single data item by index."""
        return self.data[index]

    def construct_response_text(self, data_item: Dict[str, Any]) -> str:
        """
        Build the complete response text with thinking and final answer sections.
        
        Args:
            data_item: Dictionary containing guided_solution and final_response
            
        Returns:
            Formatted response string with thinking and final response sections
        """
        response_format = '## Thinking\n\n{}\n\n## Final Response\n\n{}'
        return response_format.format(
            data_item['guided_solution'], 
            data_item['final_response']
        )

    def prepare_training_example(self, data_item: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        Convert a single data item into tokenized training example with labels.
        
        Creates input_ids and labels where only the assistant response is trained on
        (user prompt is masked with -100 in labels).
        
        Args:
            data_item: Raw data item with question and answer
            
        Returns:
            Dictionary with input_ids and labels tensors
        """
        question_text = data_item['Question']
        answer_text = self.construct_response_text(data_item)
        assert question_text is not None and answer_text is not None, \
            f'question:{question_text} answer:{answer_text}'

        # Create full conversation with question and answer
        full_conversation = self.template.render(
            messages=[
                {"role": "user", "content": question_text},
                {"role": "assistant", "content": answer_text}
            ],
            bos_token=self.tokenizer.bos_token,
            add_generation_prompt=False
        )
        conversation_ids = self.tokenizer.encode(full_conversation, add_special_tokens=False)

        # Create query-only version to determine where to mask labels
        query_only = self.template.render(
            messages=[{"role": "user", "content": question_text}],
            bos_token=self.tokenizer.bos_token,
            add_generation_prompt=True
        )
        query_only_ids = self.tokenizer.encode(query_only, add_special_tokens=False)

        # Build labels: mask prompt tokens with -100, keep response tokens
        label_sequence = [-100] * len(query_only_ids) + conversation_ids[len(query_only_ids):]
        assert len(label_sequence) == len(conversation_ids)
        
        return {
            "input_ids": conversation_ids[-self.max_seq_len:], 
            "labels": label_sequence[-self.max_seq_len:]
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate multiple examples into a batch with padding.
        
        Args:
            batch: List of data items
            
        Returns:
            Dictionary with padded input_ids and labels tensors
        """
        processed_examples = [self.prepare_training_example(item) for item in batch]
        
        input_id_sequences = [example["input_ids"] for example in processed_examples]
        label_sequences = [example["labels"] for example in processed_examples]
        
        # Determine maximum length in batch
        max_length_in_batch = max(len(sequence) for sequence in input_id_sequences)
        max_length_in_batch = min(max_length_in_batch, self.max_seq_len)
        
        # Pad sequences to max length
        padded_input_ids = [
            seq[:max_length_in_batch] + [self.tokenizer.eos_token_id] * (max_length_in_batch - len(seq))
            for seq in input_id_sequences
        ]
        padded_labels = [
            seq[:max_length_in_batch] + [-100] * (max_length_in_batch - len(seq))
            for seq in label_sequences
        ]
        
        # Debug printing for first few batches
        if self.debug < 3:
            print('input_ids', self.tokenizer.decode(padded_input_ids[-1]))
            print('labels', self.tokenizer.decode([0 if x == -100 else x for x in padded_labels[-1]]))
            self.debug += 1

        return {
            "input_ids": torch.LongTensor(padded_input_ids),
            "labels": torch.LongTensor(padded_labels),
        }
    
    def __len__(self) -> int:
        """Return the total number of training examples."""
        return len(self.data)


class TrainingMetricsTracker:
    """
    Tracks training metrics including loss and token-level accuracy across distributed processes.
    """
    
    def __init__(self, device: torch.device):
        """
        Initialize metric tracking tensors.
        
        Args:
            device: CUDA device for tensor operations
        """
        self.step_count = 0
        self.correct_predictions = torch.Tensor([0]).to(device=device)
        self.total_predictions = torch.Tensor([0]).to(device=device)
        self.accumulated_loss = torch.Tensor([0]).to(device=device)
        self.world_size = dist.get_world_size()

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor):
        """Allow instance to be called directly."""
        return self.update(logits, labels, loss)

    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor):
        """
        Update metrics with predictions from current batch.
        
        Args:
            logits: Model output logits
            labels: Ground truth labels
            loss: Computed loss value
        """
        self.step_count += 1
        
        with torch.no_grad():
            # Shift predictions and labels for next-token prediction
            shifted_predictions = logits[..., :-1, :].argmax(dim=-1)
            shifted_labels = labels[..., 1:]
            
            # Count correct predictions (excluding padding tokens)
            self.correct_predictions += (shifted_predictions == shifted_labels)\
                .masked_fill(shifted_labels.eq(-100), 0)\
                .sum()\
                .item()
            self.total_predictions += (shifted_labels != -100).sum().item()
            self.accumulated_loss += loss.item()

    def get_metric(self, reset: bool = True) -> tuple:
        """
        Aggregate metrics across all distributed processes.
        
        Args:
            reset: Whether to reset metrics after retrieval
            
        Returns:
            Tuple of (accuracy, average_loss)
        """
        # Synchronize metrics across all GPUs
        dist.all_reduce(self.correct_predictions, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total_predictions, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.accumulated_loss, op=torch.distributed.ReduceOp.SUM)

        accuracy = (self.correct_predictions / self.total_predictions).item()
        average_loss = self.accumulated_loss.item() / (self.world_size * self.step_count)

        if reset:
            self.step_count = 0
            self.correct_predictions.fill_(0)
            self.total_predictions.fill_(0)
            self.accumulated_loss.fill_(0)
            
        return accuracy, average_loss


def execute_training_pipeline(args: argparse.Namespace):
    """
    Main training execution function with distributed setup.
    
    Args:
        args: Parsed command-line arguments containing all training configuration
    """
    # Initialize Accelerator for distributed training
    accelerator = Accelerator(
        mixed_precision='bf16', 
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    # Initialize WandB logging on main process only
    if accelerator.is_main_process:
        wandb.init(
            project=args.experiment_name, 
            config=args, 
            dir=args.log_dir, 
            mode="offline"
        )
    
    accelerator.print(f'args:\n{args}')

    # Configure DeepSpeed batch size settings
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = \
        args.train_bsz_per_gpu * dist.get_world_size() * accelerator.gradient_accumulation_steps

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)

    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()

    # Configure optimizer with weight decay (excluding bias and LayerNorm)
    no_weight_decay_params = ["bias", "LayerNorm.weight"]
    optimizer_parameter_groups = [
        {
            "params": [
                param for name, param in model.named_parameters() 
                if not any(nd in name for nd in no_weight_decay_params)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                param for name, param in model.named_parameters() 
                if any(nd in name for nd in no_weight_decay_params)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_parameter_groups, lr=args.learning_rate)

    # Prepare dataset and dataloader
    training_dataset = MedicalReasoningDataset(args, tokenizer)
    training_dataloader = DataLoader(
        training_dataset, 
        batch_size=args.train_bsz_per_gpu, 
        shuffle=True, 
        drop_last=True, 
        collate_fn=training_dataset.collate_fn
    )

    # Calculate total training steps for learning rate scheduling
    total_training_steps = int(len(training_dataloader) * args.n_epochs) \
        // accelerator.gradient_accumulation_steps \
        // dist.get_world_size()
    
    learning_rate_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(args.warmup_rates * total_training_steps), 
        num_training_steps=total_training_steps
    )
    
    accelerator.print(
        f'gradient_accumulation_steps:{accelerator.gradient_accumulation_steps} '
        f'data_path:{args.data_path} lr:{args.learning_rate} '
        f'num_training_steps:{total_training_steps}'
    )
    
    # Prepare model, optimizer, and dataloader with Accelerator
    model, optimizer, training_dataloader = accelerator.prepare(
        model, optimizer, training_dataloader
    )

    # Training state tracking
    starting_epoch = 0
    starting_step = 0
    global_step_counter = 0

    # Initialize metrics tracker
    metrics_tracker = TrainingMetricsTracker(device=torch.cuda.current_device())

    def save_training_checkpoint(epoch: int, step: int, global_step: int):
        """
        Save model checkpoint with proper handling for DeepSpeed ZeRO-3.
        
        Args:
            epoch: Current epoch number
            step: Current step within epoch
            global_step: Global step counter across all epochs
        """
        checkpoint_directory = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
        
        if accelerator.is_main_process:
            # Manage maximum number of checkpoints
            existing_checkpoints = os.listdir(args.output_dir)
            existing_checkpoints = [
                file for file in existing_checkpoints 
                if file.startswith("checkpoint-")
            ]
            checkpoint_count = len(existing_checkpoints)
            
            if args.max_ckpts > 0:
                if checkpoint_count >= args.max_ckpts:
                    # Remove oldest checkpoint
                    existing_checkpoints.sort(
                        key=lambda x: os.path.getctime(os.path.join(args.output_dir, x))
                    )
                    oldest_checkpoint_path = existing_checkpoints[0]
                    shutil.rmtree(os.path.join(args.output_dir, oldest_checkpoint_path))
            
            os.makedirs(checkpoint_directory, exist_ok=True)
            model_output_directory = os.path.join(checkpoint_directory, 'tfmr')
            
            # Save model based on DeepSpeed ZeRO stage
            if accelerator.state.deepspeed_plugin.zero_stage != 3:
                model.save_pretrained(
                    model_output_directory,
                    state_dict=accelerator.get_state_dict(model)
                )
            tokenizer.save_pretrained(model_output_directory)
            
            # Copy additional files from original model directory
            copied_files = []
            for item in os.listdir(args.model_path):
                if os.path.exists(os.path.join(model_output_directory, item)):
                    continue
                if item.startswith("pytorch_model") and item.endswith(".bin"):
                    continue
                if item.endswith(".index.json") or item.endswith(".safetensors"):
                    continue
                    
                source_path = os.path.join(args.model_path, item)
                if os.path.isfile(source_path):
                    shutil.copy(source_path, os.path.join(model_output_directory, item))
                copied_files.append(item)
            
            print(f'huggingface model save in {model_output_directory}, copy file:{copied_files}')

        # Special handling for DeepSpeed ZeRO-3
        if accelerator.state.deepspeed_plugin.zero_stage == 3:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                os.path.join(checkpoint_directory, f'tfmr'),
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model)
            )
            
        # Synchronize all processes before saving training state
        accelerator.wait_for_everyone()
        accelerator.save(
            {"epoch": epoch, "step": step, "global_step": global_step}, 
            os.path.join(checkpoint_directory, "training_state.pt")
        )
        accelerator.print(f'checkpoint checkpoint-{epoch}-{global_step} is saved...')

    # Print DeepSpeed configuration
    accelerator.print(accelerator.deepspeed_config)
    model.train()

    # Main training loop
    for epoch in range(starting_epoch, args.n_epochs):
        # Create progress bar for main process only
        dataloader_iterator = tqdm(
            enumerate(training_dataloader), 
            total=len(training_dataloader)
        ) if accelerator.is_main_process else enumerate(training_dataloader)
        
        for batch_index, batch_data in dataloader_iterator:
            # Skip steps if resuming from checkpoint
            if epoch == starting_epoch and batch_index < starting_step:
                continue

            # Clear cache after first batch
            if batch_index == 1 and epoch == 0:
                torch.cuda.empty_cache()

            batch_input_ids = batch_data['input_ids']
            batch_labels = batch_data['labels']

            # Forward pass
            model_output = model(
                input_ids=batch_input_ids, 
                labels=batch_labels, 
                return_dict=True,
                use_cache=False
            )
            batch_loss = model_output.loss

            # Update metrics
            metrics_tracker(model_output.logits, batch_labels, batch_loss)
            token_accuracy, training_loss = metrics_tracker.get_metric()
            
            # Backward pass
            accelerator.backward(batch_loss)
            
            # Optimizer step (with gradient accumulation)
            if (global_step_counter + 1) % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                learning_rate_scheduler.step()
                optimizer.zero_grad()

            global_step_counter += 1

            # Update progress bar on main process
            if accelerator.is_main_process:
                dataloader_iterator.set_postfix(
                    epoch=epoch, 
                    current_step=batch_index, 
                    total_step=len(training_dataloader), 
                    skip=accelerator.optimizer_step_was_skipped, 
                    loss=round(training_loss, 3), 
                    acc=round(token_accuracy, 3), 
                    length=len(batch_input_ids[0]), 
                    lr=learning_rate_scheduler.get_last_lr()[0]
                )

            # Log to WandB periodically
            if global_step_counter % 3 == 0 and accelerator.is_main_process:
                wandb.log({
                    'skip': int(accelerator.optimizer_step_was_skipped),
                    'loss': training_loss,
                    'acc': token_accuracy,
                    'lr': learning_rate_scheduler.get_last_lr()[0]
                }, step=global_step_counter)

        # Save checkpoint at end of each epoch
        accelerator.wait_for_everyone()
        save_training_checkpoint(epoch, batch_index, global_step_counter)


if __name__ == '__main__':
    # Parse command-line arguments
    argument_parser = argparse.ArgumentParser(description='Args of sft')
    
    # Experiment configuration
    argument_parser.add_argument('--experiment_name', type=str, default='finetuning')

    # Model configuration
    argument_parser.add_argument('--model_path', required=True, type=str)

    # Data configuration
    argument_parser.add_argument('--data_path', required=True, type=str)

    # Training configuration
    argument_parser.add_argument('--output_dir', default='./ckpts', type=str)
    argument_parser.add_argument('--max_ckpts', default=2, type=int)
    argument_parser.add_argument('--log_dir', default='./train_logs', type=str)
    argument_parser.add_argument('--max_seq_len', default=8192, type=int)
    argument_parser.add_argument('--gradient_checkpointing', action='store_true')
    argument_parser.add_argument('--gradient_accumulation_steps', default=8, type=int)
    argument_parser.add_argument('--train_bsz_per_gpu', default=2, type=int)
    argument_parser.add_argument('--weight_decay', default=0.1, type=float)
    argument_parser.add_argument('--learning_rate', default=5e-6, type=float)
    argument_parser.add_argument('--warmup_rates', default=0.05, type=float)
    argument_parser.add_argument('--n_epochs', default=3, type=int)

    # Other configuration
    argument_parser.add_argument('--seed', default=42, type=int)

    training_args = argument_parser.parse_args()
    
    # Setup output directories
    training_args.log_dir = os.path.join(training_args.log_dir, training_args.experiment_name)
    training_args.output_dir = os.path.join(training_args.output_dir, training_args.experiment_name)

    os.makedirs(training_args.log_dir, exist_ok=True)
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Set random seed for reproducibility
    set_seed(training_args.seed)
    
    # Execute training
    execute_training_pipeline(training_args)