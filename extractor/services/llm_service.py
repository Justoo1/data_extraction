import json
import logging
import subprocess
from django.conf import settings

logger = logging.getLogger(__name__)

class LLMService:
    """Service for interacting with Ollama running DeepSeek model for data extraction using subprocess"""
    
    def __init__(self, model=None):
        """
        Initialize the LLM service
        
        Args:
            model: The model to use (default: from settings.DEEPSEEK_MODEL or 'deepseek-r1:14b-qwen-distill-q4_K_M')
        """
        self.model = model or getattr(settings, 'DEEPSEEK_MODEL', 'deepseek-r1:14b-qwen-distill-q4_K_M')
        
    def _validate_installation(self):
        """Validate that Ollama is installed and the model is available"""
        try:
            # Check if Ollama is installed
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to run Ollama: {result.stderr}")
                return False
            
            # Check if the model is available
            if self.model not in result.stdout:
                logger.warning(f"Model {self.model} not found in available models")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Failed to validate Ollama installation: {str(e)}")
            return False
    
    def generate(self, prompt, max_tokens=4096, temperature=0.2):
        """
        Generate a response using Ollama via subprocess
        
        Args:
            prompt: The prompt to send to the model
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation (0-1)
            
        Returns:
            The generated text
        """
        try:
            if not self._validate_installation():
                raise Exception("Could not connect to Ollama or model not available")
            
            # Build Ollama command with parameters
            # Note: ollama run doesn't support the same options as the API,
            # so we'll add parameters to the prompt as a system instruction
            system_instruction = f"""
            Temperature: {temperature}
            Max tokens: {max_tokens}
            
            Respond only with the content requested, without any preamble or additional explanation.
            """
            
            full_prompt = f"{system_instruction}\n\n{prompt}"
            
            # Run Ollama using subprocess with explicit encoding control
            process = subprocess.Popen(
                ["ollama", "run", self.model],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Send input as UTF-8 encoded bytes
            stdout, stderr = process.communicate(full_prompt.encode('utf-8'))
            
            # Decode output with UTF-8 and handle any decoding errors
            output = stdout.decode('utf-8', errors='replace')
            error_output = stderr.decode('utf-8', errors='replace')
            
            if process.returncode != 0:
                logger.error(f"Ollama process failed: {error_output}")
                raise Exception(f"Ollama process failed: {error_output}")
            
            # Return the output
            return output.strip()
            
        except Exception as e:
            logger.error(f"Error generating response from LLM: {str(e)}")
            raise
    
    def detect_sendig_domains(self, text, domain_descriptions):
        """
        Detect SENDIG domains in text
        
        Args:
            text: The text to analyze
            domain_descriptions: Dictionary mapping domain codes to descriptions
            
        Returns:
            List of detected domains with confidence scores and page ranges
        """
        prompt = f"""
        You are an expert in SENDIG (Standard for Exchange of Nonclinical Data Implementation Guide) data standards.
        Analyze the following text from a toxicology report and identify which SENDIG domains are present.
        
        Here are the SENDIG domains to consider:
        {json.dumps(domain_descriptions, indent=2)}
        
        For each domain you identify, provide:
        1. The domain code
        2. A confidence score between 0 and 1
        3. An explanation of why you believe this domain is present
        
        Format your response as valid JSON with the following structure:
        {{
            "domains": [
                {{
                    "code": "domain_code",
                    "confidence": confidence_score,
                    "explanation": "explanation"
                }}
            ]
        }}
        
        Text to analyze:
        {text[:5000]}  # Truncate to prevent token limit issues
        """
        
        try:
            response = self.generate(prompt, temperature=0.1)
            # Extract JSON from response
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_text = response[json_start:json_end]
                    return json.loads(json_text)
                else:
                    # If we can't find JSON markers, try parsing the whole response
                    return json.loads(response)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from LLM response: {response}")
                return {"domains": []}
        except Exception as e:
            logger.error(f"Error detecting domains: {str(e)}")
            return {"domains": []}
    
    def extract_structured_data(self, text, domain_code, required_variables):
        """
        Extract structured data for a specific SENDIG domain
        
        Args:
            text: The text to extract data from
            domain_code: The SENDIG domain code
            required_variables: List of required variables for this domain
            
        Returns:
            Structured data in JSON format
        """
        prompt = f"""
        You are an expert in extracting SENDIG (Standard for Exchange of Nonclinical Data Implementation Guide) data.
        Extract structured data for the {domain_code} domain from the following text.
        
        The {domain_code} domain requires the following variables:
        {json.dumps(required_variables, indent=2)}
        
        For each record you identify, extract values for these variables.
        If a value is not present or cannot be determined, use null.
        
        Format your response as valid JSON with the following structure:
        {{
            "records": [
                {{
                    "variable1": "value1",
                    "variable2": "value2",
                    ...
                }}
            ]
        }}
        
        Text to analyze:
        {text[:8000]}  # Truncate to prevent token limit issues
        """
        
        try:
            response = self.generate(prompt, max_tokens=8192, temperature=0.1)  # Fixed token limit to reasonable value
            # Extract JSON from response
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_text = response[json_start:json_end]
                    return json.loads(json_text)
                else:
                    # If we can't find JSON markers, try parsing the whole response
                    return json.loads(response)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from LLM response: {response}")
                return {"records": []}
        except Exception as e:
            logger.error(f"Error extracting structured data: {str(e)}")
            return {"records": []}