"""
Naive Chain-of-Thought Fine-Tuning Pipeline
Alternative training approach using simple CoT responses without guided refinement.
Used for ablation study comparing baseline CoT against feedback-guided methods.
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

# HuggingFace token should be set as environment variable
# Run: export HF_TOKEN="your_token_here"
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# Initialize logging configuration
logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')


class NaiveCoTDataset(torch.utils.data.Dataset):
    """
    Dataset for naive Chain-of-Thought training data.
    Processes question-response pairs without guided refinement structure.
    """
    
    def __init__(self, config: argparse.Namespace, tokenizer: AutoTokenizer):
        """
        Initialize dataset with configuration and tokenizer.
        
        Args:
            config: Training configuration with data path and sequence settings
            tokenizer: Pre-trained tokenizer for text encoding
        """
        self.config = config
        self.tokenizer = tokenizer
        
        # Load naive CoT training data
        with open(config.data_path) as data_file:
            self.data = json.load(data_file)
        
        # Apply any necessary filtering
        filtered_examples = []
        for example in self.data:
            filtered_examples.append(example)
        print('Filtered out', len(self.data), len(filtered_examples))
        self.data = filtered_examples

        self.max_seq_len = self.config.max_seq_len
        self.debug = 0

        # Using Llama3-instruct chat template for base model training
        llama3_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        
        # Set chat template if not already configured
        if not tokenizer.chat_template:
            tokenizer.chat_template = llama3_template
            
        self.template = Template(tokenizer.chat_template)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Retrieve single training example by index."""
        return self.data[index]

    def extract_response_text(self, example: Dict[str, Any]) -> str:
        """
        Extract response text from data item.
        For naive CoT data, response is in single 'response' field.
        
        Args:
            example: Data dictionary containing response
            
        Returns:
            Response text string
        """
        return example['response']

    def prepare_training_prompt(self, example: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        Convert data item into tokenized training example.
        Note: Uses lowercase 'question' field for naive CoT format.
        
        Args:
            example: Raw data item with question and response
            
        Returns:
            Dictionary with input_ids and labels
        """
        question_text = example['question']
        response_text = self.extract_response_text(example)
        assert question_text is not None and response_text is not None, \
            f'question:{question_text} response:{response_text}'

        # Build complete conversation sequence
        full_input = self.template.render(
            messages=[
                {"role": "user", "content": question_text},
                {"role": "assistant", "content": response_text}
            ],
            bos_token=self.tokenizer.bos_token,
            add_generation_prompt=False
        )
        input_token_ids = self.tokenizer.encode(full_input, add_special_tokens=False)

        # Build query-only sequence to identify prompt boundary
        query_sequence = self.template.render(
            messages=[{"role": "user", "content": question_text}],
            bos_token=self.tokenizer.bos_token,
            add_generation_prompt=True
        )
        query_token_ids = self.tokenizer.encode(query_sequence, add_special_tokens=False)

        # Create label sequence (mask prompt tokens with -100)
        label_ids = [-100] * len(query_token_ids) + input_token_ids[len(query_token_ids):]
        assert len(label_ids) == len(input_token_ids)
        
        return {
            "input_ids": input_token_ids[-self.max_seq_len:], 
            "labels": label_ids[-self.max_seq_len:]
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of examples with appropriate padding.
        
        Args:
            batch: List of raw data items
            
        Returns:
            Dictionary with batched and padded tensors
        """
        processed_batch = [self.prepare_training_prompt(item) for item in batch]
        
        input_sequences = [item["input_ids"] for item in processed_batch]
        label_sequences = [item["labels"] for item in processed_batch]
        
        # Determine maximum sequence length in batch
        max_batch_length = max(len(seq) for seq in input_sequences)
        max_batch_length = min(max_batch_length, self.max_seq_len)
        
        # Pad all sequences to maximum length
        padded_inputs = [
            seq[:max_batch_length] + [self.tokenizer.eos_token_id] * (max_batch_length - len(seq))
            for seq in input_sequences
        ]
        padded_labels = [
            seq[:max_batch_length] + [-100] * (max_batch_length - len(seq))
            for seq in label_sequences
        ]
        
        # Debug output for first few batches
        if self.debug < 3:
            print('input_ids', self.tokenizer.decode(padded_inputs[-1]))
            print('labels', self.tokenizer.decode([0 if x == -100 else x for x in padded_labels[-1]]))
            self.debug += 1

        return {
            "input_ids": torch.LongTensor(padded_inputs),
            "labels": torch.LongTensor(padded_labels),
        }
    
    def __len__(self) -> int:
        """Return total number of training examples."""
        return len(self.data)


class AccuracyMetricsCalculator:
    """
    Tracks and aggregates training metrics across distributed processes.
    Computes token-level accuracy and loss during training.
    """
    
    def __init__(self, device: torch.device):
        """
        Initialize metric tracking tensors on specified device.
        
        Args:
            device: CUDA device for tensor operations
        """
        self.step_count = 0
        self.correct_token_count = torch.Tensor([0]).to(device=device)
        self.total_token_count = torch.Tensor([0]).to(device=device)
        self.loss_accumulator = torch.Tensor([0]).to(device=device)
        self.world_size = dist.get_world_size()

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor):
        """Enable direct calling of instance."""
        return self.update(logits, labels, loss)

    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor):
        """
        Update metrics with current batch predictions.
        
        Args:
            logits: Model output logits
            labels: Ground truth token labels
            loss: Computed loss tensor
        """
        self.step_count += 1
        
        with torch.no_grad():
            # Compute next-token predictions
            predicted_tokens = logits[..., :-1, :].argmax(dim=-1)
            target_tokens = labels[..., 1:]
            
            # Count correct predictions (excluding padding)
            self.correct_token_count += (predicted_tokens == target_tokens)\
                .masked_fill(target_tokens.eq(-100), 0)\
                .sum()\
                .item()
            self.total_token_count += (target_tokens != -100).sum().item()
            self.loss_accumulator += loss.item()

    def get_metric(self, reset: bool = True) -> tuple:
        """
        Aggregate and return metrics across all processes.
        
        Args:
            reset: Whether to reset counters after retrieval
            
        Returns:
            Tuple of (token_accuracy, average_loss)
        """
        # Synchronize metrics across distributed processes
        dist.all_reduce(self.correct_token_count, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total_token_count, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.loss_accumulator, op=torch.distributed.ReduceOp.SUM)

        token_accuracy = (self.correct_token_count / self.total_token_count).item()
        mean_loss = self.loss_accumulator.item() / (self.world_size * self.step_count)

        if reset:
            self.step_count = 0
            self.correct_token_count.fill_(0)
            self.total_token_count.fill_(0)
            self.loss_accumulator.fill_(0)
            
        return token_accuracy, mean_loss


def run_training_loop(args: argparse.Namespace):
    """
    Execute main training loop for naive CoT fine-tuning.
    
    Args:
        args: Parsed command-line arguments with training configuration
    """
    # Initialize distributed training accelerator
    accelerator = Accelerator(
        mixed_precision='bf16', 
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    # Initialize experiment tracking on main process
    if accelerator.is_main_process:
        wandb.init(
            project=args.experiment_name, 
            config=args, 
            dir=args.log_dir, 
            mode="offline"
        )
    
    accelerator.print(f'args:\n{args}')

    # Configure DeepSpeed batch size parameters
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = \
        args.train_bsz_per_gpu * dist.get_world_size() * accelerator.gradient_accumulation_steps

    # Load tokenizer and language model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    language_model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)

    # Enable gradient checkpointing for memory efficiency
    language_model.gradient_checkpointing_enable()

    # Configure optimizer parameter groups with weight decay
    params_without_decay = ["bias", "LayerNorm.weight"]
    optimizer_params = [
        {
            "params": [
                p for n, p in language_model.named_parameters() 
                if not any(nd in n for nd in params_without_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in language_model.named_parameters() 
                if any(nd in n for nd in params_without_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_params, lr=args.learning_rate)

    # Prepare dataset and dataloader
    train_data = NaiveCoTDataset(args, tokenizer)
    data_loader = DataLoader(
        train_data, 
        batch_size=args.train_bsz_per_gpu, 
        shuffle=True, 
        drop_last=True, 
        collate_fn=train_data.collate_fn
    )

    # Calculate total training steps for scheduler
    total_steps = int(len(data_loader) * args.n_epochs) \
        // accelerator.gradient_accumulation_steps \
        // dist.get_world_size()
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(args.warmup_rates * total_steps), 
        num_training_steps=total_steps
    )
    
    accelerator.print(
        f'gradient_accumulation_steps:{accelerator.gradient_accumulation_steps} '
        f'data_path:{args.data_path} lr:{args.learning_rate} '
        f'num_training_steps:{total_steps}'
    )
    
    # Prepare model and optimizer with accelerator
    language_model, optimizer, data_loader = accelerator.prepare(
        language_model, optimizer, data_loader
    )

    # Initialize training state variables
    initial_epoch = 0
    initial_step = 0
    global_iteration = 0

    # Initialize metrics tracker
    metric_calculator = AccuracyMetricsCalculator(device=torch.cuda.current_device())

    def persist_checkpoint(epoch_num: int, step_num: int, global_step_num: int):
        """
        Save training checkpoint with model weights and state.
        
        Args:
            epoch_num: Current epoch number
            step_num: Current step within epoch
            global_step_num: Global step counter
        """
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{epoch_num}-{global_step_num}")
        
        if accelerator.is_main_process:
            # Manage checkpoint count limit
            existing_checkpoints = os.listdir(args.output_dir)
            existing_checkpoints = [
                f for f in existing_checkpoints 
                if f.startswith("checkpoint-")
            ]
            num_existing = len(existing_checkpoints)
            
            if args.max_ckpts > 0:
                if num_existing >= args.max_ckpts:
                    # Remove oldest checkpoint
                    existing_checkpoints.sort(
                        key=lambda x: os.path.getctime(os.path.join(args.output_dir, x))
                    )
                    oldest_ckpt = existing_checkpoints[0]
                    shutil.rmtree(os.path.join(args.output_dir, oldest_ckpt))
            
            os.makedirs(checkpoint_path, exist_ok=True)
            transformer_output_path = os.path.join(checkpoint_path, 'tfmr')
            
            # Save model based on DeepSpeed configuration
            if accelerator.state.deepspeed_plugin.zero_stage != 3:
                language_model.save_pretrained(
                    transformer_output_path,
                    state_dict=accelerator.get_state_dict(language_model)
                )
            tokenizer.save_pretrained(transformer_output_path)
            
            # Copy supplementary files from base model
            files_copied = []
            for filename in os.listdir(args.model_path):
                if os.path.exists(os.path.join(transformer_output_path, filename)):
                    continue
                if filename.startswith("pytorch_model") and filename.endswith(".bin"):
                    continue
                if filename.endswith(".index.json") or filename.endswith(".safetensors"):
                    continue
                    
                source_file = os.path.join(args.model_path, filename)
                if os.path.isfile(source_file):
                    shutil.copy(source_file, os.path.join(transformer_output_path, filename))
                files_copied.append(filename)
            
            print(f'huggingface model save in {transformer_output_path}, copy file:{files_copied}')

        # Handle DeepSpeed ZeRO-3 checkpoint saving
        if accelerator.state.deepspeed_plugin.zero_stage == 3:
            unwrapped_model = accelerator.unwrap_model(language_model)
            unwrapped_model.save_pretrained(
                os.path.join(checkpoint_path, f'tfmr'),
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(language_model)
            )
            
        # Synchronize processes before saving state
        accelerator.wait_for_everyone()
        accelerator.save(
            {"epoch": epoch_num, "step": step_num, "global_step": global_step_num}, 
            os.path.join(checkpoint_path, "training_state.pt")
        )
        accelerator.print(f'checkpoint checkpoint-{epoch_num}-{global_step_num} is saved...')

    # Display DeepSpeed configuration
    accelerator.print(accelerator.deepspeed_config)
    language_model.train()

    # Main training loop across epochs
    for current_epoch in range(initial_epoch, args.n_epochs):
        # Setup progress tracking for main process
        loader_iterator = tqdm(
            enumerate(data_loader), 
            total=len(data_loader)
        ) if accelerator.is_main_process else enumerate(data_loader)
        
        for batch_idx, batch_data in loader_iterator:
            # Skip batches when resuming from checkpoint
            if current_epoch == initial_epoch and batch_idx < initial_step:
                continue

            # Clear GPU cache after first batch
            if batch_idx == 1 and current_epoch == 0:
                torch.cuda.empty_cache()

            batch_inputs = batch_data['input_ids']
            batch_labels = batch_data['labels']

            # Forward pass through model
            model_output = language_model(
                input_ids=batch_inputs, 
                labels=batch_labels, 
                return_dict=True,
                use_cache=False
            )
            computed_loss = model_output.loss

            # Update and retrieve metrics
            metric_calculator(model_output.logits, batch_labels, computed_loss)
            current_accuracy, current_loss = metric_calculator.get_metric()
            
            # Backward pass
            accelerator.backward(computed_loss)
            
            # Perform optimizer step with gradient accumulation
            if (global_iteration + 1) % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_iteration += 1

            # Update progress display on main process
            if accelerator.is_main_process:
                loader_iterator.set_postfix(
                    epoch=current_epoch, 
                    current_step=batch_idx, 
                    total_step=len(data_loader), 
                    skip=accelerator.optimizer_step_was_skipped, 
                    loss=round(current_loss, 3), 
                    acc=round(current_accuracy, 3), 
                    length=len(batch_inputs[0]), 
                    lr=lr_scheduler.get_last_lr()[0]
                )

            # Log metrics to WandB periodically
            if global_iteration % 3 == 0 and accelerator.is_main_process:
                wandb.log({
                    'skip': int(accelerator.optimizer_step_was_skipped),
                    'loss': current_loss,
                    'acc': current_accuracy,
                    'lr': lr_scheduler.get_last_lr()[0]
                }, step=global_iteration)

        # Save checkpoint at end of each epoch
        accelerator.wait_for_everyone()
        persist_checkpoint(current_epoch, batch_idx, global_iteration)


if __name__ == '__main__':
    # Configure command-line argument parser
    arg_parser = argparse.ArgumentParser(description='Args of naive cot finetuning')
    
    # Experiment configuration
    arg_parser.add_argument('--experiment_name', type=str, default='naive_cot_finetuning')

    # Model configuration
    arg_parser.add_argument('--model_path', required=True, type=str)

    # Data configuration
    arg_parser.add_argument('--data_path', required=True, type=str)

    # Training configuration
    arg_parser.add_argument('--output_dir', default='./ckpts', type=str)
    arg_parser.add_argument('--max_ckpts', default=2, type=int)
    arg_parser.add_argument('--log_dir', default='./train_logs', type=str)
    arg_parser.add_argument('--max_seq_len', default=8192, type=int)
    arg_parser.add_argument('--gradient_checkpointing', action='store_true')
    arg_parser.add_argument('--gradient_accumulation_steps', default=8, type=int)
    arg_parser.add_argument('--train_bsz_per_gpu', default=2, type=int)
    arg_parser.add_argument('--weight_decay', default=0.1, type=float)
    arg_parser.add_argument('--learning_rate', default=5e-6, type=float)
    arg_parser.add_argument('--warmup_rates', default=0.05, type=float)
    arg_parser.add_argument('--n_epochs', default=3, type=int)

    # Other configuration
    arg_parser.add_argument('--seed', default=42, type=int)

    parsed_args = arg_parser.parse_args()
    
    # Configure output directories
    parsed_args.log_dir = os.path.join(parsed_args.log_dir, parsed_args.experiment_name)
    parsed_args.output_dir = os.path.join(parsed_args.output_dir, parsed_args.experiment_name)

    os.makedirs(parsed_args.log_dir, exist_ok=True)
    os.makedirs(parsed_args.output_dir, exist_ok=True)

    # Set random seed for reproducibility
    set_seed(parsed_args.seed)
    
    # Execute training
    run_training_loop(parsed_args)