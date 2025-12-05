"""
Encoder-decoder model for code documentation generation (CodeT5)
Specialized for code-to-text and text-to-code tasks
"""

import torch
from transformers import (
    T5ForConditionalGeneration,
    RobertaTokenizer,
    GenerationConfig
)
from typing import List, Dict, Optional

class CodeT5DocGenerator:
    """Wrapper for CodeT5-based documentation generation"""
    
    def __init__(
        self,
        model_name: str = "Salesforce/codet5-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model_name = model_name
        
        print(f"Loading {model_name}...")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        
        print(f"Model loaded successfully on {device}")
    
    def prepare_inputs(
        self,
        code: str,
        documentation: Optional[str] = None,
        max_source_length: int = 512,
        max_target_length: int = 128
    ) -> Dict:
        """Prepare inputs for training/inference"""
        
        # For CodeT5, we can use task prefixes
        # Format: "generate documentation: <code>"
        source_text = f"generate documentation: {code}"
        
        inputs = self.tokenizer(
            source_text,
            max_length=max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        model_inputs = {
            'input_ids': inputs['input_ids'].to(self.device),
            'attention_mask': inputs['attention_mask'].to(self.device)
        }
        
        if documentation is not None:
            # Prepare labels for training
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    documentation,
                    max_length=max_target_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
            
            # Replace padding token id with -100 so it's ignored in loss
            labels['input_ids'][labels['input_ids'] == self.tokenizer.pad_token_id] = -100
            model_inputs['labels'] = labels['input_ids'].to(self.device)
        
        return model_inputs
    
    def generate(
        self,
        code: str,
        max_length: int = 128,
        num_beams: int = 5,
        temperature: float = 1.0,
        top_p: float = 0.95,
        do_sample: bool = False,
        **kwargs
    ) -> str:
        """Generate documentation for given code"""
        
        inputs = self.prepare_inputs(code)
        
        generation_config = GenerationConfig(
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            early_stopping=True,
            **kwargs
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                generation_config=generation_config
            )
        
        documentation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return documentation
    
    def batch_generate(
        self,
        codes: List[str],
        batch_size: int = 8,
        **kwargs
    ) -> List[str]:
        """Generate documentation for multiple code snippets"""
        
        documentations = []
        
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i:i+batch_size]
            
            # Prepare batch inputs
            source_texts = [f"generate documentation: {code}" for code in batch_codes]
            inputs = self.tokenizer(
                source_texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    **kwargs
                )
            
            batch_docs = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
            documentations.extend(batch_docs)
        
        return documentations
    
    def train_step(self, code: str, documentation: str) -> float:
        """Single training step"""
        self.model.train()
        
        inputs = self.prepare_inputs(code, documentation)
        
        outputs = self.model(**inputs)
        loss = outputs.loss
        
        return loss.item()
    
    def save(self, output_dir: str):
        """Save model and tokenizer"""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
    
    @classmethod
    def load(cls, model_dir: str, device: str = None):
        """Load saved model"""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        return cls(model_name=model_dir, device=device)


class CodeT5WithLoRA:
    """CodeT5 with LoRA for parameter-efficient fine-tuning"""
    
    def __init__(
        self,
        base_model: str = "Salesforce/codet5-base",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        from peft import LoraConfig, get_peft_model, TaskType
        
        self.device = device
        self.base_model_name = base_model
        
        # Load base model
        self.tokenizer = RobertaTokenizer.from_pretrained(base_model)
        self.model = T5ForConditionalGeneration.from_pretrained(base_model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q", "v"],  # T5 attention modules
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.to(device)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
    
    def generate(self, code: str, **kwargs) -> str:
        """Generate with LoRA model"""
        generator = CodeT5DocGenerator.__new__(CodeT5DocGenerator)
        generator.model = self.model
        generator.tokenizer = self.tokenizer
        generator.device = self.device
        generator.model_name = self.base_model_name
        
        return generator.generate(code, **kwargs)


class CodeT5Trainer:
    """Training utilities for CodeT5"""
    
    def __init__(
        self,
        model: CodeT5DocGenerator,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    def train_epoch(self, train_dataloader, device: str = "cuda"):
        """Train for one epoch"""
        self.model.model.train()
        total_loss = 0
        
        from tqdm import tqdm
        for batch in tqdm(train_dataloader, desc="Training"):
            # Forward pass
            outputs = self.model.model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return total_loss / len(train_dataloader)


if __name__ == "__main__":
    # Demo usage
    print("Initializing CodeT5 for documentation generation...")
    generator = CodeT5DocGenerator()
    
    sample_code = """
class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        if not self.root:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)
"""
    
    print("\nSample Code:")
    print(sample_code)
    
    print("\nGenerating documentation...")
    doc = generator.generate(sample_code, max_length=128, num_beams=5)
    print(f"\nGenerated Documentation:\n{doc}")
