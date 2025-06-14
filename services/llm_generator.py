from langchain_community.llms import LlamaCpp
import os
import re

class LLMGenerator:
    def __init__(self, model_path="./services/unsloth.Q8_0.gguf"):
        self.model_path = model_path
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLaMA model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.llm = LlamaCpp(
                model_path=self.model_path,
                temperature=0.3,  # Lower temperature for more consistent code generation
                max_tokens=512,   # Increased for longer configurations
                top_p=0.95,
                n_ctx=4096,      # Increased context window
                n_threads=8,
                verbose=False,   # Set to False to reduce noise
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLaMA model: {str(e)}")
    
    def generate_nix_config(self, prompt):
        """Generate NixOS configuration based on user prompt"""
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        
        try:
            # Use the prompt directly since the model is fine-tuned
            response = self.llm(prompt)
            # Strip common prefixes and clean the response
            return self._clean_response(response)
        except Exception as e:
            raise RuntimeError(f"Failed to generate configuration: {str(e)}")
    
    def _clean_response(self, response):
        """Clean the LLM response by detecting and extracting the actual Nix code"""
        cleaned = response.strip()
        
        # Find the last }; in the string and remove everything after it
        last_brace = cleaned.rfind('};') + 1
        if last_brace != -1:
            cleaned = cleaned[:last_brace + 1]
        
        # Look for patterns like "word =" or "word.word =" to find the start of Nix code
        lines = cleaned.split('\n')
        start_line = 0
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            # Look for Nix attribute patterns: word = or word.word.word =
            # This matches things like "programs.fontconfig =" or "services.nginx ="
            if '=' in stripped_line:
                # Find the part before the =
                before_equals = stripped_line.split('=')[0].strip()
                # Check if it looks like a Nix attribute (contains only letters, dots, numbers, underscores)
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_.-]*$', before_equals):
                    start_line = i
                    break
        
        # If we found a valid start line, remove everything before it
        if start_line > 0:
            cleaned = '\n'.join(lines[start_line:])
        cleaned = "{\n" + cleaned + "\n}"
        
        return cleaned.strip()
    
    def is_model_loaded(self):
        """Check if the model is properly loaded"""
        return self.llm is not None
