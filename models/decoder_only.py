"""
Decoder-only model for code documentation generation (CodeLlama)
Uses autoregressive generation with prompt engineering
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)
from typing import Optional, List, Dict

class CodeLlamaDocGenerator:
    """Wrapper for CodeLlama-based documentation generation"""
    
    PROMPT_TEMPLATE = """# Task: Generate comprehensive documentation for the following code

# Code:
```{language}
{code}
```

# Documentation:
"""
    
    def __init__(
        self,
        model_name: str = "codellama/CodeLlama-7b-hf",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False
    ):
        self.device = device
        self.model_name = model_name
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with optional quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            load_in_8bit=load_in_8bit
        )
        
        if not load_in_8bit and device == "cuda":
            self.model.to(device)
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded successfully on {device}")
    
    def create_prompt(self, code: str, language: str = "python") -> str:
        """Create prompt for documentation generation"""
        return self.PROMPT_TEMPLATE.format(language=language, code=code)
    
    def generate(
        self,
        code: str,
        language: str = "python",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_beams: int = 1,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """Generate documentation for given code"""
        
        prompt = self.create_prompt(code, language)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # Decode and extract only the generated documentation
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract documentation part (after "# Documentation:")
        if "# Documentation:" in full_output:
            documentation = full_output.split("# Documentation:")[1].strip()
        else:
            documentation = full_output[len(prompt):].strip()
        
        return documentation
    
    def batch_generate(
        self,
        codes: List[str],
        languages: List[str] = None,
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """Generate documentation for multiple code snippets"""
        
        if languages is None:
            languages = ["python"] * len(codes)
        
        documentations = []
        
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i:i+batch_size]
            batch_langs = languages[i:i+batch_size]
            
            prompts = [
                self.create_prompt(code, lang)
                for code, lang in zip(batch_codes, batch_langs)
            ]
            
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **kwargs)
            
            batch_docs = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
            # Extract documentation parts
            for doc in batch_docs:
                if "# Documentation:" in doc:
                    documentations.append(doc.split("# Documentation:")[1].strip())
                else:
                    documentations.append(doc.strip())
        
        return documentations
    
    def save(self, output_dir: str):
        """Save model and tokenizer"""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
    
    @classmethod
    def load(cls, model_dir: str, device: str = None, load_in_8bit: bool = False):
        """Load saved model"""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        return cls(model_name=model_dir, device=device, load_in_8bit=load_in_8bit)


class CodeLlamaWithLoRA:
    """CodeLlama with LoRA for parameter-efficient fine-tuning"""
    
    def __init__(
        self,
        base_model: str = "codellama/CodeLlama-7b-hf",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        from peft import LoraConfig, get_peft_model, TaskType
        
        self.device = device
        self.base_model_name = base_model
        
        # Load base model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
    
    def generate(self, code: str, **kwargs) -> str:
        """Generate with LoRA model (same interface as base)"""
        generator = CodeLlamaDocGenerator.__new__(CodeLlamaDocGenerator)
        generator.model = self.model
        generator.tokenizer = self.tokenizer
        generator.device = self.device
        generator.model_name = self.base_model_name
        
        return generator.generate(code, **kwargs)


if __name__ == "__main__":
    # Demo usage
    print("Initializing CodeLlama for documentation generation...")
    
    # Use smaller model for demo or set load_in_8bit=True
    generator = CodeLlamaDocGenerator(
        model_name="codellama/CodeLlama-7b-hf",
        load_in_8bit=False  # Set to True to reduce memory
    )
    
    sample_code = """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
"""
    
    print("\nSample Code:")
    print(sample_code)
    
    print("\nGenerating documentation...")
    doc = generator.generate(sample_code, max_new_tokens=200, temperature=0.7)
    print(f"\nGenerated Documentation:\n{doc}")
