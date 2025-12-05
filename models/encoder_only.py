"""
Encoder-only model for code documentation generation (CodeBERT)
Uses a decoder head on top of the encoder for generation
"""

import torch
from torch import nn
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    RobertaConfig,
    PreTrainedModel
)
from typing import Optional, Dict

class CodeBERTForDocGeneration(PreTrainedModel):
    """
    CodeBERT with a decoder head for documentation generation.
    Uses encoder representations to generate documentation autoregressively.
    """
    
    config_class = RobertaConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        
        # Generation head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict:
        """Forward pass"""
        
        # Get encoder outputs
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token representation for generation
        sequence_output = outputs.last_hidden_state
        
        # Generate logits
        logits = self.lm_head(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if outputs.hidden_states else None,
            'attentions': outputs.attentions if outputs.attentions else None
        }
    
    def generate_documentation(
        self,
        tokenizer,
        code_input_ids: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 4,
        **kwargs
    ) -> str:
        """Generate documentation for given code"""
        
        self.eval()
        with torch.no_grad():
            # Simple greedy generation (in practice, use beam search)
            generated_ids = []
            current_ids = code_input_ids
            
            for _ in range(max_length):
                outputs = self.forward(current_ids)
                next_token_logits = outputs['logits'][:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1)
                
                generated_ids.append(next_token_id.item())
                
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
                
                current_ids = torch.cat([current_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
        
        return tokenizer.decode(generated_ids, skip_special_tokens=True)


class CodeBERTDocGenerator:
    """Wrapper class for CodeBERT documentation generation"""
    
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # Load base model
        config = RobertaConfig.from_pretrained(model_name)
        self.model = CodeBERTForDocGeneration(config)
        
        # Load pretrained weights for encoder
        pretrained = RobertaModel.from_pretrained(model_name)
        self.model.roberta = pretrained
        
        self.model.to(device)
    
    def prepare_inputs(self, code: str, documentation: str = None):
        """Prepare inputs for training/inference"""
        
        # Tokenize code
        code_inputs = self.tokenizer(
            code,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        inputs = {
            'input_ids': code_inputs['input_ids'].to(self.device),
            'attention_mask': code_inputs['attention_mask'].to(self.device)
        }
        
        if documentation:
            # Tokenize documentation for training
            doc_inputs = self.tokenizer(
                documentation,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            inputs['labels'] = doc_inputs['input_ids'].to(self.device)
        
        return inputs
    
    def generate(self, code: str, **kwargs) -> str:
        """Generate documentation for code"""
        self.model.eval()
        inputs = self.prepare_inputs(code)
        
        with torch.no_grad():
            # Use the model's generate method
            documentation = self.model.generate_documentation(
                self.tokenizer,
                inputs['input_ids'],
                **kwargs
            )
        
        return documentation
    
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
        
        instance = cls.__new__(cls)
        instance.device = device
        instance.tokenizer = RobertaTokenizer.from_pretrained(model_dir)
        instance.model = CodeBERTForDocGeneration.from_pretrained(model_dir)
        instance.model.to(device)
        
        return instance


if __name__ == "__main__":
    # Demo usage
    print("Initializing CodeBERT for documentation generation...")
    generator = CodeBERTDocGenerator()
    
    sample_code = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
"""
    
    print("\nSample Code:")
    print(sample_code)
    
    print("\nGenerating documentation...")
    doc = generator.generate(sample_code, max_length=50)
    print(f"\nGenerated Documentation:\n{doc}")
