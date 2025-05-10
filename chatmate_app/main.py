#!/usr/bin/env python3
import sys
import json
import os
import sqlite3
import random
from datetime import datetime
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QLineEdit, 
                            QListWidget, QListWidgetItem, QTabWidget, QTextEdit,
                            QScrollArea, QFrame, QSplitter, QMessageBox, QComboBox,
                            QDialog, QCheckBox, QSizePolicy, QSlider, QColorDialog,
                            QFileDialog, QProgressBar, QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QColor, QIcon, QPalette, QPixmap, QImage, QPainter, QBrush, QLinearGradient
import subprocess

# Constants
TODO_FILE = "todos.json"
DB_FILE = "chatmate.db"
CATEGORIES = ["To Do", "Ongoing", "Done", "Waiting", "Someday"]

# AI Moods
AI_MOODS = {
    "professional": {
        "name": "Professional",
        "description": "Formal and business-like responses",
        "color": "#4a90e2",
        "system_prompt": "You are a professional assistant providing clear, accurate information in a formal tone."
    },
    "friendly": {
        "name": "Friendly",
        "description": "Warm and conversational responses",
        "color": "#50c878",
        "system_prompt": "You are a friendly assistant having a casual conversation. Be warm, approachable and occasionally use emojis."
    },
    "creative": {
        "name": "Creative",
        "description": "Imaginative and artistic responses",
        "color": "#9370db",
        "system_prompt": "You are a creative assistant with an artistic flair. Use metaphors, vivid descriptions, and think outside the box."
    },
    "concise": {
        "name": "Concise",
        "description": "Brief and to-the-point responses",
        "color": "#ff7f50",
        "system_prompt": "You are a concise assistant. Provide brief, direct answers with minimal elaboration."
    },
    "technical": {
        "name": "Technical",
        "description": "Detailed technical responses",
        "color": "#607d8b",
        "system_prompt": "You are a technical assistant with deep expertise. Provide detailed, technical explanations with precise terminology."
    }
}

# Theme definitions
THEMES = {
    "dark": {
        "app_bg": "#121212",
        "sidebar_bg": "#1E1E1E",
        "chat_bg": "#121212",
        "input_bg": "#1E2428",
        "user_bubble": "#128C7E",
        "ai_bubble": "#262D31",
        "system_bubble": "#333333",
        "text": "#FFFFFF",
        "secondary_text": "#AAAAAA",
        "border": "#2D383E",
        "button": "#00A884",
        "button_hover": "#128C7E",
        "error": "#FF6B6B"
    },
    "light": {
        "app_bg": "#F0F2F5",
        "sidebar_bg": "#FFFFFF",
        "chat_bg": "#E4DDD6",
        "input_bg": "#FFFFFF",
        "user_bubble": "#D9FDD3",
        "ai_bubble": "#FFFFFF",
        "system_bubble": "#ECECEC",
        "text": "#111B21",
        "secondary_text": "#667781",
        "border": "#D1D7DB",
        "button": "#00A884",
        "button_hover": "#008C7E",
        "error": "#F15C6D"
    },
    "blue": {
        "app_bg": "#0A1929",
        "sidebar_bg": "#0A1929",
        "chat_bg": "#0A1929",
        "input_bg": "#132F4C",
        "user_bubble": "#0059B2",
        "ai_bubble": "#132F4C",
        "system_bubble": "#1E4976",
        "text": "#FFFFFF",
        "secondary_text": "#AAC7E4",
        "border": "#1E4976",
        "button": "#007FFF",
        "button_hover": "#0059B2",
        "error": "#EB0014"
    }
}

class OnlineAIAPI:
    """Class to handle communication with online AI APIs"""
    
    def __init__(self):
        self.api_key = ""
        self.api_endpoint = ""
        self.available = False
        self.provider = "none"
        self.models = []
        self.current_model = ""
        
    def set_api_key(self, api_key, provider="openai"):
        """Set the API key and provider"""
        self.api_key = api_key
        self.provider = provider.lower()
        
        # Set the appropriate endpoint based on provider
        if self.provider == "openai":
            self.api_endpoint = "https://api.openai.com/v1/chat/completions"
            self.models = ["gpt-3.5-turbo", "gpt-4"]
            self.current_model = "gpt-3.5-turbo"
            self.available = True
        elif self.provider == "anthropic":
            self.api_endpoint = "https://api.anthropic.com/v1/messages"
            self.models = ["claude-instant", "claude-2"]
            self.current_model = "claude-instant"
            self.available = True
        elif self.provider == "huggingface":
            self.api_endpoint = "https://api-inference.huggingface.co/models/"
            self.models = ["google/flan-t5-xxl", "facebook/bart-large-cnn"]
            self.current_model = "google/flan-t5-xxl"
            self.available = True
        else:
            self.available = False
            
        return self.available
    
    def set_model(self, model_name):
        """Set the model to use"""
        if model_name in self.models:
            self.current_model = model_name
            return True
        return False
    
    def generate_response(self, prompt, system_prompt="You are a helpful AI assistant"):
        """Generate a response from the online API"""
        import requests
        import json
        
        if not self.available or not self.api_key:
            return "Error: No API key set or service unavailable. Please configure an API key first."
        
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            if self.provider == "openai":
                headers["Authorization"] = f"Bearer {self.api_key}"
                data = {
                    "model": self.current_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7
                }
                
                response = requests.post(self.api_endpoint, headers=headers, json=data, timeout=30)
                
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    return f"Error: {response.status_code} - {response.text}"
                    
            elif self.provider == "anthropic":
                headers["x-api-key"] = self.api_key
                headers["anthropic-version"] = "2023-06-01"
                
                data = {
                    "model": self.current_model,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000
                }
                
                response = requests.post(self.api_endpoint, headers=headers, json=data, timeout=30)
                
                if response.status_code == 200:
                    return response.json()["content"][0]["text"]
                else:
                    return f"Error: {response.status_code} - {response.text}"
                    
            elif self.provider == "huggingface":
                headers["Authorization"] = f"Bearer {self.api_key}"
                
                full_url = f"{self.api_endpoint}{self.current_model}"
                data = {"inputs": f"{system_prompt}\n\n{prompt}"}
                
                response = requests.post(full_url, headers=headers, json=data, timeout=30)
                
                if response.status_code == 200:
                    return response.json()[0]["generated_text"]
                else:
                    return f"Error: {response.status_code} - {response.text}"
            
            return "Error: Unsupported provider"
            
        except Exception as e:
            return f"Error connecting to {self.provider.capitalize()} API: {str(e)}"

class ChatbotAPI:
    """Class to handle communication with Ollama API"""
    
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        # Default models if none are found
        self.models = ["llama3.2", "phi3", "mistral", "codellama", "gemma2"]
        self.available_models = []
        self.current_model_index = 0
        self.model = self.models[self.current_model_index] if self.models else None
        self.offline_mode = False
        self.model_sizes = {}
        self.model_capabilities = {}  # Store capabilities for each model
        self.online_api = OnlineAIAPI()
        self.use_online_api = False
        self.current_mood = "professional"  # Default mood
        
        # Models known to support image generation
        self.image_capable_models = [
            "llava", "bakllava", "llava-llama3", "llava-phi3", "moondream",
            "cogvlm", "dall-e", "stable-diffusion", "midjourney", "sdxl",
            "pixart", "kandinsky", "playground", "deepfloyd", "imagen"
        ]
        
        self.local_model_paths = [
            "/home/udaysingh/.ollama/models",  # Default Ollama path
            "/usr/local/share/ollama/models",  # System-wide Ollama models
            os.path.expanduser("~/models"),  # User's models directory
            os.path.expanduser("~/.local/share/models")  # Alternative location
        ]
        self.get_available_models()
        
    def get_available_models(self):
        """Get list of available models from Ollama and local directories"""
        import subprocess
        import json
        import glob
        import os
        
        # First try to get models from Ollama API
        ollama_models_found = self.get_ollama_models()
        
        # If no Ollama models found, scan local directories for model files
        if not ollama_models_found or not self.available_models:
            self.scan_local_model_directories()
        
        # If we have models, set the current one
        if self.available_models:
            self.models = self.available_models[:10]  # Take up to 10 models
            self.model = self.models[0]
            self.current_model_index = 0
            self.offline_mode = False
            return True
        else:
            # No models found, stay with defaults
            return False
    
    def get_ollama_models(self):
        """Get models from Ollama API"""
        import subprocess
        import json
        
        try:
            # Try to run ollama list and capture output
            result = subprocess.run(["ollama", "list", "--json"], 
                                   capture_output=True, text=True, check=True)
            
            if result.returncode == 0 and result.stdout:
                # Parse JSON output
                try:
                    models_data = json.loads(result.stdout)
                    # Extract model names and sizes
                    self.available_models = []
                    self.model_sizes = {}
                    self.model_capabilities = {}
                    
                    for model in models_data:
                        name = model.get('name', '').split(':')[0]
                        size = model.get('size', 0)
                        self.model_sizes[name] = size
                        self.available_models.append(name)
                        
                        # Check if model supports image generation
                        supports_images = any(img_model in name.lower() for img_model in self.image_capable_models)
                        self.model_capabilities[name] = {
                            "image_generation": supports_images,
                            "size_mb": size / (1024 * 1024) if size > 0 else 0
                        }
                    
                    # Get more detailed model info for each model
                    self.get_model_details()
                    
                    # Sort models by size (smallest first)
                    self.available_models.sort(key=lambda x: self.model_sizes.get(x, float('inf')))
                    return True
                except json.JSONDecodeError:
                    # Fallback to text parsing if JSON fails
                    return self.parse_ollama_list_text(result.stdout)
            else:
                return self.check_ollama_status()
                
        except (subprocess.SubprocessError, FileNotFoundError):
            return self.check_ollama_status()
            
    def get_model_details(self):
        """Get detailed information about each model"""
        import subprocess
        import json
        import time
        
        for model_name in self.available_models:
            try:
                # Try to get model details using ollama show
                result = subprocess.run(["ollama", "show", model_name, "--json"], 
                                      capture_output=True, text=True, check=False,
                                      timeout=2)  # Short timeout to avoid hanging
                
                if result.returncode == 0 and result.stdout:
                    try:
                        model_info = json.loads(result.stdout)
                        
                        # Extract capabilities
                        if "details" in model_info:
                            # Check for multimodal capability
                            is_multimodal = "vision" in model_info.get("details", {}).get("capabilities", [])
                            if is_multimodal:
                                self.model_capabilities[model_name]["image_generation"] = True
                                
                            # Check for image generation in model description
                            model_desc = model_info.get("details", {}).get("description", "").lower()
                            if any(term in model_desc for term in ["image", "visual", "vision", "multimodal", "picture"]):
                                self.model_capabilities[model_name]["image_generation"] = True
                    except json.JSONDecodeError:
                        pass
            except (subprocess.SubprocessError, FileNotFoundError, TimeoutError):
                # Skip if we can't get details
                pass
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.1)
    
    def scan_local_model_directories(self):
        """Scan local directories for model files and query Ollama for available models"""
        import os
        import glob
        import subprocess
        import json
        import time
        
        # First, try to get all Ollama models directly using the API
        try:
            # Try direct API call to Ollama
            import requests
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models_data = response.json()
                if "models" in models_data:
                    for model in models_data["models"]:
                        model_name = model.get("name", "").split(":")[0]  # Remove version tag if present
                        if model_name and model_name not in self.available_models:
                            self.available_models.append(model_name)
                            # Check if model supports image generation based on name
                            supports_images = any(img_model in model_name.lower() for img_model in self.image_capable_models)
                            self.model_capabilities[model_name] = {
                                "image_generation": supports_images,
                                "size_mb": model.get("size", 0) / (1024 * 1024) if model.get("size", 0) > 0 else 1000,
                                "source": "ollama_api"
                            }
        except Exception as e:
            print(f"Error querying Ollama API: {e}")
        
        # Also try using the ollama CLI command
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False, timeout=5)
            if result.returncode == 0 and result.stdout:
                # Parse the output
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Skip header line
                    for line in lines[1:]:  # Skip header
                        parts = line.split()
                        if parts:  # Ensure there's at least one part
                            model_name = parts[0].split(":")[0]  # Get name without tag
                            if model_name and model_name not in self.available_models:
                                self.available_models.append(model_name)
                                # Check if model supports image generation based on name
                                supports_images = any(img_model in model_name.lower() for img_model in self.image_capable_models)
                                self.model_capabilities[model_name] = {
                                    "image_generation": supports_images,
                                    "size_mb": 1000,  # Default size
                                    "source": "ollama_cli"
                                }
        except Exception as e:
            print(f"Error running ollama list command: {e}")
        
        # Model file extensions to look for
        model_extensions = ["*.bin", "*.gguf", "*.ggml", "*.safetensors", "*.pt", "*.pth"]
        
        # Additional Ollama model paths to check
        additional_ollama_paths = [
            "/var/lib/ollama",  # System-wide Ollama installation
            os.path.expanduser("~/.ollama"),  # User's Ollama directory
            "/usr/share/ollama",  # Alternative system location
            "/opt/ollama"  # Optional installation location
        ]
        
        # Add additional paths to scan
        for path in additional_ollama_paths:
            if path not in self.local_model_paths:
                self.local_model_paths.append(path)
        
        # Now scan local directories for model files
        for path in self.local_model_paths:
            if os.path.exists(path) and os.path.isdir(path):
                # Look for model files with various extensions
                for ext in model_extensions:
                    try:
                        model_files = glob.glob(os.path.join(path, "**", ext), recursive=True)
                        
                        for model_file in model_files:
                            model_name = os.path.basename(model_file).split(".")[0]
                            if model_name not in self.available_models:
                                self.available_models.append(model_name)
                                
                                # Estimate size in MB
                                try:
                                    size_mb = os.path.getsize(model_file) / (1024 * 1024)
                                    self.model_sizes[model_name] = size_mb
                                except:
                                    size_mb = 1000  # Default size if can't determine
                                    self.model_sizes[model_name] = size_mb
                                
                                # Check if model supports image generation based on name
                                supports_images = any(img_model in model_name.lower() for img_model in self.image_capable_models)
                                
                                # Also check if model file path contains hints about image capabilities
                                file_path_lower = model_file.lower()
                                if any(term in file_path_lower for term in ["vision", "multimodal", "image", "visual"]):
                                    supports_images = True
                                    
                                self.model_capabilities[model_name] = {
                                    "image_generation": supports_images,
                                    "size_mb": size_mb,
                                    "local_path": model_file,
                                    "source": "file_scan"
                                }
                    except Exception as e:
                        print(f"Error scanning {path} for {ext}: {e}")
                        
        # Sort available models alphabetically
        self.available_models.sort()
        
        # Update models list with available models
        if self.available_models:
            self.models = self.available_models[:20]  # Limit to 20 models to avoid overwhelming the UI
    
    def parse_ollama_list_text(self, output):
        """Parse the text output of ollama list command"""
        import re
        
        try:
            # Extract model names and sizes using regex
            pattern = r'(\S+)\s+\S+\s+(\d+(?:\.\d+)?\s+\w+)'
            matches = re.findall(pattern, output)
            
            if matches:
                self.available_models = []
                self.model_sizes = {}
                
                for name, size_str in matches:
                    name = name.split(':')[0]  # Remove version tag if present
                    # Convert size to numeric value for comparison
                    size_parts = size_str.split()
                    if len(size_parts) >= 2:
                        size_value = float(size_parts[0])
                        size_unit = size_parts[1].upper()
                        
                        # Convert to MB for comparison
                        if 'GB' in size_unit:
                            size_value *= 1024
                        elif 'KB' in size_unit:
                            size_value /= 1024
                            
                        self.model_sizes[name] = size_value
                        self.available_models.append(name)
                
                # Sort models by size (smallest first)
                self.available_models.sort(key=lambda x: self.model_sizes.get(x, float('inf')))
                
                # Take the 5 smallest models
                self.models = self.available_models[:5] if self.available_models else self.models
                
                if self.models:
                    self.model = self.models[0]
                    self.current_model_index = 0
                    self.offline_mode = False
                    return True
            
            return self.check_ollama_status()
        except Exception:
            return self.check_ollama_status()
    
    def check_ollama_status(self):
        """Check if Ollama is running and set appropriate mode"""
        import requests
        import subprocess
        
        try:
            # Try to connect to Ollama API
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.offline_mode = False
                return True
            else:
                self.offline_mode = True
                return False
        except Exception:
            # If we can't connect, try to start Ollama service
            try:
                subprocess.Popen(["ollama", "serve"], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE)
                # Wait a moment for it to start
                import time
                time.sleep(3)
                # Try again
                return self.check_ollama_status()
            except:
                self.offline_mode = True
                return False
    
    def set_model(self, model_name):
        """Set the current model"""
        if model_name in self.models:
            self.model = model_name
            self.current_model_index = self.models.index(model_name)
            return True
        return False
    
    def try_next_model(self):
        """Try the next model in the list"""
        if not self.models:
            return None
            
        self.current_model_index = (self.current_model_index + 1) % len(self.models)
        self.model = self.models[self.current_model_index]
        return self.model
        
    def generate_response(self, prompt, system_prompt="You are a helpful AI assistant"):
        """Generate a response from the Ollama API or online API"""
        import requests
        import subprocess
        import json
        
        # If using online API, delegate to it
        if self.use_online_api and self.online_api.available:
            return self.online_api.generate_response(prompt, system_prompt)
        
        # If we're in offline mode, return a canned response
        if self.offline_mode:
            return self.get_offline_response(prompt)
        
        # Try direct CLI approach first (more reliable)
        try:
            # Use subprocess to run the ollama command directly
            cmd = ["ollama", "run", self.model, f"{system_prompt}\n\n{prompt}"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout:
                # Successfully got a response via CLI
                return result.stdout.strip()
            else:
                print(f"CLI Error with {self.model}: {result.stderr}")
                # Continue to API approach if CLI fails
        except Exception as e:
            print(f"Error running Ollama CLI: {str(e)}")
            # Continue to API approach if CLI fails
        
        # Try API approach as fallback
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "system": system_prompt,
                        "stream": False
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json().get("response", "Sorry, I couldn't generate a response.")
                else:
                    # If there's an error, try the next model
                    error_msg = f"Error with model {self.model}: {response.status_code} - {response.text}"
                    print(error_msg)
                    self.try_next_model()
            except Exception as e:
                error_msg = f"Error connecting to Ollama with model {self.model}: {str(e)}"
                print(error_msg)
                self.try_next_model()
        
        # If all attempts failed and we have online API configured, try that
        if self.online_api.available:
            self.use_online_api = True
            return self.online_api.generate_response(prompt, system_prompt)
        
        # If everything failed, switch to offline mode
        self.offline_mode = True
        return self.get_offline_response(prompt)
    
    def get_offline_response(self, prompt):
        """Generate a response without using Ollama"""
        import random
        
        # Fallback responses for offline mode
        fallback_responses = [
            "I'm currently in offline mode. Please check if Ollama is running.",
            "It seems I can't connect to the AI model right now. Please check your connection.",
            "I'm having trouble accessing the language model. Please try again later.",
            "The AI service appears to be unavailable at the moment. Please verify it's running.",
            "I'm unable to process your request in offline mode. Please ensure Ollama is running."
        ]
        
        return random.choice(fallback_responses)
        
    def can_generate_images(self):
        """Check if the current model can generate images"""
        if self.use_online_api:
            # Check if online API supports image generation
            return self.online_api.provider.lower() in ["openai", "anthropic", "stability", "midjourney"]
        elif self.model in self.model_capabilities:
            # Check local model capabilities
            return self.model_capabilities[self.model].get("image_generation", False)
        return False
        
    def check_model_capabilities(self, model_name):
        """Check detailed capabilities of a specific model by querying it"""
        import os
        import json
        import requests
        import time
        import traceback
        
        # Validate model name
        if not model_name:
            return {
                "success": False,
                "error": "No model name provided",
                "capabilities": {"name": "Unknown"}
            }
        
        # Handle case where model might not be in self.models but still valid
        if model_name not in self.models and model_name not in self.available_models:
            # Try to add it to available models
            self.available_models.append(model_name)
            self.models.append(model_name)
            
        # Initialize capabilities with basic info
        model_info = self.model_capabilities.get(model_name, {})
        source = model_info.get("source", "unknown")
        
        # Start with known capabilities or defaults
        capabilities = {
            "name": model_name,
            "source": source,
            "image_generation": model_info.get("image_generation", False),
            "size_mb": model_info.get("size_mb", 0),
            "check_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Flag to track if we got any meaningful data
        got_meaningful_data = False
        
        # For Ollama models, try multiple methods to get capabilities
        if source in ["ollama_api", "ollama_cli"] or source == "unknown":
            # Method 1: Try Ollama API 'show' endpoint
            try:
                response = requests.post(
                    f"{self.base_url}/api/show",
                    json={"name": model_name},
                    timeout=5
                )
                
                if response.status_code == 200:
                    model_data = response.json()
                    if model_data:
                        got_meaningful_data = True
                        # Don't store the entire details object as it can be large
                        if "parameters" in model_data:
                            capabilities["parameters"] = model_data["parameters"]
                        if "license" in model_data:
                            capabilities["license"] = model_data["license"]
                        if "template" in model_data:
                            capabilities["has_template"] = True
                        if "family" in model_data:
                            capabilities["family"] = model_data["family"]
                        
                        # Extract model card info
                        if "modelfile" in model_data:
                            modelfile = model_data["modelfile"]
                            lines = modelfile.split("\n")
                            
                            # Parse modelfile for capabilities
                            for line in lines:
                                if line.startswith("FROM "):
                                    capabilities["base_model"] = line.replace("FROM ", "").strip()
                                elif line.startswith("PARAMETER "):
                                    param_line = line.replace("PARAMETER ", "").strip()
                                    if " " in param_line:
                                        param_name, param_value = param_line.split(" ", 1)
                                        if "parameters" not in capabilities:
                                            capabilities["parameters"] = {}
                                        capabilities["parameters"][param_name] = param_value
                                elif line.startswith("TEMPLATE "):
                                    capabilities["has_template"] = True
                                elif "vision" in line.lower() or "image" in line.lower() or "multimodal" in line.lower():
                                    capabilities["image_generation"] = True
                                    capabilities["vision_capable"] = True
            except Exception as e:
                # Just log the error and continue to other methods
                print(f"Error getting model info from Ollama API: {str(e)}")
                capabilities["api_error"] = str(e)
            
            # Method 2: Try to query the model directly if Method 1 failed or didn't provide enough info
            if not got_meaningful_data or "self_reported" not in capabilities:
                try:
                    # Simple capability check prompt
                    check_prompt = "What capabilities do you have? Can you process images or generate images? Please answer briefly."
                    
                    # Use a shorter timeout to avoid hanging
                    response = requests.post(
                        f"{self.base_url}/api/chat",
                        json={
                            "model": model_name,
                            "messages": [
                                {"role": "user", "content": check_prompt}
                            ],
                            "stream": False,
                            "options": {"temperature": 0}
                        },
                        timeout=8
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if "message" in result and "content" in result["message"]:
                            got_meaningful_data = True
                            capabilities["self_reported"] = result["message"]["content"]
                            
                            # Check for vision/image capabilities in the response
                            response_text = result["message"]["content"].lower()
                            if ("vision" in response_text or "image" in response_text) and \
                               ("process" in response_text or "understand" in response_text or "analyze" in response_text):
                                capabilities["vision_capable"] = True
                            
                            if "generate image" in response_text or "create image" in response_text:
                                capabilities["image_generation"] = True
                except Exception as e:
                    # Just log the error and continue
                    print(f"Error querying model: {str(e)}")
                    capabilities["query_error"] = str(e)
        
        # For local file models, try to determine capabilities from the file path and name
        if source == "file_scan" and "local_path" in model_info:
            try:
                file_path = model_info["local_path"]
                if os.path.exists(file_path):
                    got_meaningful_data = True
                    file_name = os.path.basename(file_path).lower()
                    
                    # Check for vision/image capabilities in the file name or path
                    if any(term in file_name or term in file_path.lower() for term in 
                          ["vision", "vl", "visual", "multimodal", "multi-modal", "image"]):
                        capabilities["vision_capable"] = True
                        capabilities["image_generation"] = True
                    
                    # Try to determine model type from file extension
                    file_ext = os.path.splitext(file_path)[1].lower()
                    if file_ext in [".gguf", ".ggml"]:
                        capabilities["model_type"] = "llama.cpp compatible"
                    elif file_ext in [".safetensors"]:
                        capabilities["model_type"] = "Hugging Face compatible"
                    elif file_ext in [".pt", ".pth"]:
                        capabilities["model_type"] = "PyTorch model"
                    elif file_ext in [".bin"]:
                        capabilities["model_type"] = "Binary model"
                    
                    # Try to determine model size from file size
                    try:
                        size_bytes = os.path.getsize(file_path)
                        capabilities["size_mb"] = size_bytes / (1024 * 1024)
                        capabilities["size_gb"] = size_bytes / (1024 * 1024 * 1024)
                    except Exception as size_error:
                        print(f"Error getting file size: {str(size_error)}")
            except Exception as e:
                print(f"Error checking file model: {str(e)}")
        
        # If we didn't get any meaningful data, try to infer from model name
        if not got_meaningful_data:
            # Infer capabilities from model name
            model_lower = model_name.lower()
            
            # Check for common model families
            if any(family in model_lower for family in ["llama", "mistral", "phi", "gemma", "gpt"]):
                capabilities["model_family"] = "Large Language Model"
            
            # Check for vision capabilities in name
            if any(term in model_lower for term in ["vision", "vl", "visual", "multimodal", "multi-modal"]):
                capabilities["vision_capable"] = True
                capabilities["image_generation"] = True
        
        # Update the model_capabilities with the new information
        try:
            if model_name not in self.model_capabilities:
                self.model_capabilities[model_name] = {}
                
            self.model_capabilities[model_name].update({
                k: v for k, v in capabilities.items() 
                if k not in ["name"] and v is not None
            })
        except Exception as update_error:
            print(f"Error updating model capabilities: {str(update_error)}")
        
        return {
            "success": True,
            "capabilities": capabilities
        }
        
    def generate_image(self, prompt, size="512x512"):
        """Generate an image based on the prompt"""
        import requests
        import json
        import tempfile
        import os
        import base64
        from datetime import datetime
        
        if not self.can_generate_images():
            return None, "Current model does not support image generation"
            
        try:
            if self.use_online_api:
                # Use online API for image generation
                if self.online_api.provider.lower() == "openai":
                    # OpenAI DALL-E
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.online_api.api_key}"
                    }
                    data = {
                        "model": "dall-e-3",
                        "prompt": prompt,
                        "size": size,
                        "n": 1
                    }
                    response = requests.post(
                        "https://api.openai.com/v1/images/generations",
                        headers=headers,
                        json=data
                    )
                    if response.status_code == 200:
                        result = response.json()
                        image_url = result["data"][0]["url"]
                        # Download the image
                        img_response = requests.get(image_url)
                        if img_response.status_code == 200:
                            # Save to temp file
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            img_path = os.path.join(tempfile.gettempdir(), f"generated_image_{timestamp}.png")
                            with open(img_path, "wb") as f:
                                f.write(img_response.content)
                            return img_path, None
                    return None, f"Error: {response.text}"
                    
                # Add support for other online providers here
                return None, "Image generation not implemented for this provider"
            else:
                # Use local Ollama model for image generation
                if not self.offline_mode:
                    headers = {"Content-Type": "application/json"}
                    data = {
                        "model": self.model,
                        "prompt": f"Generate an image of: {prompt}",
                        "stream": False
                    }
                    
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        headers=headers,
                        json=data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        response_text = result.get("response", "")
                        
                        # Check if response contains a base64 image
                        if "data:image/" in response_text and ";base64," in response_text:
                            # Extract base64 data
                            img_data = response_text.split(";base64,")[1].strip()
                            img_bytes = base64.b64decode(img_data)
                            
                            # Save to temp file
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            img_path = os.path.join(tempfile.gettempdir(), f"generated_image_{timestamp}.png")
                            with open(img_path, "wb") as f:
                                f.write(img_bytes)
                            return img_path, None
                        
                        return None, "Model did not return an image"
                    
                    return None, f"Error: {response.text if hasattr(response, 'text') else 'Unknown error'}"
                
                return None, "Cannot generate images in offline mode"
        except Exception as e:
            return None, f"Error generating image: {str(e)}"
        # Default response
        return "I'm currently in offline mode. The Ollama AI service isn't available right now. You can still use all the todo list features, but AI assistance is limited."

class TodoItem(QWidget):
    """Custom widget for todo items"""
    deleted = pyqtSignal(object)
    status_changed = pyqtSignal(object, str)
    
    def __init__(self, todo, parent=None):
        super().__init__(parent)
        self.todo = todo
        
        # Main layout with more space for content
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins to allow more content space
        
        # Task text
        self.task_label = QLabel(todo["task"])
        self.task_label.setWordWrap(True)
        
        # Status combo box
        self.status_combo = QComboBox()
        self.status_combo.addItems(CATEGORIES)
        self.status_combo.setCurrentText(todo["status"])
        self.status_combo.currentTextChanged.connect(self.on_status_changed)
        
        # Delete button
        delete_btn = QPushButton("×")
        delete_btn.setFixedSize(30, 30)
        delete_btn.clicked.connect(self.on_delete_clicked)
        
        # Add widgets to layout
        layout.addWidget(self.task_label, 1)
        layout.addWidget(self.status_combo)
        layout.addWidget(delete_btn)
        
        # Style - code editor theme
        self.setStyleSheet("""
            QWidget {
                background-color: #2d2d30;
                border-radius: 3px;
                border-left: 3px solid #007acc;
                margin: 2px 0px;
            }
            QLabel {
                font-size: 14px;
                font-family: 'Consolas', 'Monaco', monospace;
                color: #d4d4d4;
                padding-left: 5px;
            }
            QPushButton {
                background-color: #333333;
                color: #d4d4d4;
                border: none;
                border-radius: 3px;
                font-weight: bold;
                font-family: 'Consolas', 'Monaco', monospace;
            }
            QPushButton:hover {
                background-color: #c75050;
                color: #ffffff;
            }
            QComboBox {
                background-color: #3c3c3c;
                color: #d4d4d4;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                padding: 2px 5px;
                font-family: 'Consolas', 'Monaco', monospace;
            }
        """)
        
    def on_delete_clicked(self):
        self.deleted.emit(self.todo)
        
    def on_status_changed(self, new_status):
        self.todo["status"] = new_status
        self.status_changed.emit(self.todo, new_status)

class OnlineAPIDialog(QDialog):
    """Dialog for configuring online AI API settings"""
    
    def __init__(self, online_api=None, parent=None):
        super().__init__(parent)
        self.online_api = online_api
        self.api_key = ""
        self.provider = "OpenAI"
        self.init_ui()
        
    def init_ui(self):
        """Initialize the dialog UI"""
        self.setWindowTitle("Configure Online AI API")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # Provider selection
        provider_layout = QHBoxLayout()
        provider_label = QLabel("Provider:")
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["OpenAI", "Anthropic", "HuggingFace"])
        self.provider_combo.currentTextChanged.connect(self.update_model_list)
        
        provider_layout.addWidget(provider_label)
        provider_layout.addWidget(self.provider_combo)
        
        # API Key input
        key_layout = QHBoxLayout()
        key_label = QLabel("API Key:")
        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.Password)
        self.key_input.setPlaceholderText("Enter your API key here")
        
        key_layout.addWidget(key_label)
        key_layout.addWidget(self.key_input)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        
        # Update model list based on initial provider
        self.update_model_list(self.provider_combo.currentText())
        
        # Use online API checkbox
        self.use_online_checkbox = QCheckBox("Use online API instead of local models")
        self.use_online_checkbox.setChecked(True)
        
        # Set initial values if online_api is provided
        if self.online_api:
            # Set the provider
            index = self.provider_combo.findText(self.online_api.provider.capitalize())
            if index >= 0:
                self.provider_combo.setCurrentIndex(index)
                
            # Set the API key
            if self.online_api.api_key:
                self.key_input.setText(self.online_api.api_key)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_settings)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        
        # Add all layouts to main layout
        layout.addLayout(provider_layout)
        layout.addLayout(key_layout)
        layout.addLayout(model_layout)
        layout.addWidget(self.use_online_checkbox)
        layout.addLayout(button_layout)
        
        # Apply styles
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QLabel {
                color: #d4d4d4;
            }
            QLineEdit {
                background-color: #252526;
                color: #d4d4d4;
                border: 1px solid #3c3c3c;
                padding: 5px;
                border-radius: 2px;
            }
            QComboBox {
                background-color: #252526;
                color: #d4d4d4;
                border: 1px solid #3c3c3c;
                padding: 5px;
                border-radius: 2px;
            }
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 2px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QCheckBox {
                color: #d4d4d4;
            }
        """)
        
    def update_model_list(self, provider):
        """Update the model list based on selected provider"""
        self.model_combo.clear()
        
        if provider.lower() == "openai":
            self.model_combo.addItems(["gpt-3.5-turbo", "gpt-4"])
        elif provider.lower() == "anthropic":
            self.model_combo.addItems(["claude-instant", "claude-2"])
        elif provider.lower() == "huggingface":
            self.model_combo.addItems(["google/flan-t5-xxl", "facebook/bart-large-cnn"])
    
    def save_settings(self):
        """Save the API settings"""
        if not self.online_api:
            self.reject()
            return
            
        api_key = self.key_input.text().strip()
        provider = self.provider_combo.currentText()
        model = self.model_combo.currentText()
        use_online = self.use_online_checkbox.isChecked()
        
        if not api_key:
            QMessageBox.warning(self, "Warning", "API key cannot be empty")
            return
        
        # Store the values for the parent dialog to access
        self.api_key = api_key
        self.provider = provider.lower()
        
        # Try to validate the API key
        try:
            # Simple validation - just check if the key isn't empty
            if api_key:
                self.accept()
            else:
                QMessageBox.critical(self, "Error", f"Failed to configure {provider} API")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error: {str(e)}")

class MessageBubble(QWidget):
    """Custom widget for message bubbles"""
    
    def __init__(self, content, is_user=True, parent=None):
        super().__init__(parent)
        self.content = content
        self.is_user = is_user
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        # Check if dark mode is enabled
        self.dark_mode = self.is_dark_mode()
        self.init_ui()
        
    def is_dark_mode(self):
        """Check if the application is in dark mode by looking at the palette"""
        app = QApplication.instance()
        if app:
            palette = app.palette()
            bg_color = palette.color(QPalette.Window)
            # If background color is dark, we're in dark mode
            return bg_color.lightness() < 128
        return False
        
    def init_ui(self):
        """Initialize the UI"""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Header with avatar and name
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 0, 12, 0)
        header_layout.setSpacing(8)
        
        # Create avatar label (circle with initials or icon)
        avatar = QLabel()
        avatar.setFixedSize(24, 24)
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setStyleSheet("font-weight: bold; color: white; border-radius: 12px;")
        
        # Create name label
        name_label = QLabel()
        name_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        
        if self.is_user:
            # User avatar - blue circle with "U"
            avatar.setText("U")
            avatar.setStyleSheet("background-color: #128C7E; color: white; border-radius: 12px; font-weight: bold;")
            name_label.setText("You")
            name_label.setStyleSheet("color: #128C7E; font-weight: bold; font-size: 13px;")
            header_layout.setAlignment(Qt.AlignRight)
            header_layout.addWidget(name_label)
            header_layout.addWidget(avatar)
        else:
            # AI avatar - purple circle with "AI"
            avatar.setText("AI")
            avatar.setStyleSheet("background-color: #9370DB; color: white; border-radius: 12px; font-weight: bold; font-size: 10px;")
            name_label.setText("ChatMate Assistant")
            name_label.setStyleSheet("color: #9370DB; font-weight: bold; font-size: 13px;")
            header_layout.addWidget(avatar)
            header_layout.addWidget(name_label)
            header_layout.addStretch(1)
        
        # Message container with rounded corners
        container = QFrame()
        container.setObjectName("messageContainer")
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        # Container layout
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(10, 6, 10, 6)  # Thinner margins
        
        # Avatar label
        avatar_label = QLabel()
        avatar_size = 32
        avatar_label.setFixedSize(avatar_size, avatar_size)
        avatar_label.setScaledContents(True)
        
        # Set avatar image based on sender
        if self.is_user:
            avatar_pixmap = QPixmap("user_avatar.png")
            if not avatar_pixmap.isNull():
                avatar_label.setPixmap(avatar_pixmap)
            else:
                # Fallback to text avatar
                avatar_label.setText("")
                avatar_label.setAlignment(Qt.AlignCenter)
                if self.dark_mode:
                    avatar_label.setStyleSheet("font-size: 20px; background-color: #2A2A2A; border-radius: 16px;")
                else:
                    avatar_label.setStyleSheet("font-size: 20px; background-color: #E0E0E0; border-radius: 16px;")
        else:
            avatar_pixmap = QPixmap("ai_avatar.png")
            if not avatar_pixmap.isNull():
                avatar_label.setPixmap(avatar_pixmap)
            else:
                # Fallback to text avatar
                avatar_label.setText("")
                avatar_label.setAlignment(Qt.AlignCenter)
                if self.dark_mode:
                    avatar_label.setStyleSheet("font-size: 20px; background-color: #2A2A2A; border-radius: 16px;")
                else:
                    avatar_label.setStyleSheet("font-size: 20px; background-color: #E0E0E0; border-radius: 16px;")
        
        # Message content
        message_widget = QWidget()
        message_layout = QVBoxLayout(message_widget)
        message_layout.setContentsMargins(0, 0, 0, 0)
        
        # Sender name
        sender_label = QLabel("You" if self.is_user else "ChatMate")
        # Set sender color based on dark mode
        if self.dark_mode:
            sender_color = "#BBDEFB" if self.is_user else "#E0E0E0"
        else:
            sender_color = "#555555"
        sender_label.setStyleSheet(f"font-weight: bold; font-size: 12px; color: {sender_color};")
        message_layout.addWidget(sender_label)
        
        # Message text
        message_label = QLabel(self.content)
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        message_layout.addWidget(message_label)
        
        # Add avatar and message to container
        if self.is_user:
            container_layout.addWidget(message_widget)
            container_layout.addWidget(avatar_label)
        else:
            container_layout.addWidget(avatar_label)
            container_layout.addWidget(message_widget)
        
        # Style based on sender and dark mode
        if self.is_user:
            if self.dark_mode:
                container.setStyleSheet(
                    "#messageContainer { "
                    "background-color: #1A3C5E; "
                    "border-radius: 12px; "
                    "margin-left: 60px; "
                    "margin-right: 10px; "
                    "border: 0.5px solid #2C5F8E; "
                    "} "
                    "QLabel { "
                    "color: #FFFFFF; "
                    "}"
                )
            else:
                container.setStyleSheet(
                    "#messageContainer { "
                    "background-color: #E3F2FD; "
                    "border-radius: 12px; "
                    "margin-left: 60px; "
                    "margin-right: 10px; "
                    "border: 0.5px solid #BBDEFB; "
                    "} "
                    "QLabel { "
                    "color: #0D47A1; "
                    "}"
                )
        else:
            if self.dark_mode:
                container.setStyleSheet(
                    "#messageContainer { "
                    "background-color: #2A2A2A; "
                    "border-radius: 12px; "
                    "margin-left: 10px; "
                    "margin-right: 60px; "
                    "border: 0.5px solid #3A3A3A; "
                    "} "
                    "QLabel { "
                    "color: #FFFFFF; "
                    "}"
                )
            else:
                container.setStyleSheet(
                    "#messageContainer { "
                    "background-color: #F5F5F5; "
                    "border-radius: 12px; "
                    "margin-left: 10px; "
                    "margin-right: 60px; "
                    "border: 0.5px solid #E0E0E0; "
                    "} "
                    "QLabel { "
                    "color: #212121; "
                    "}"
                )
        
        layout.addWidget(container)

class SystemMessageBubble(QWidget):
    """Custom widget for system message bubbles"""
    
    def __init__(self, message, is_error=False, parent=None):
        super().__init__(parent)
        self.message = message
        self.is_error = is_error
        self.is_user = False  # System messages are never from the user
        # Check if dark mode is enabled
        self.dark_mode = self.is_dark_mode()
        self.init_ui()
    
    def is_dark_mode(self):
        """Check if the application is in dark mode by looking at the palette"""
        app = QApplication.instance()
        if app:
            palette = app.palette()
            bg_color = palette.color(QPalette.Window)
            # If background color is dark, we're in dark mode
            return bg_color.lightness() < 128
        return False
        
    def init_ui(self):
        """Initialize the UI"""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Message container with rounded corners
        container = QFrame()
        container.setObjectName("messageContainer")
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        # Container layout
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(10, 6, 10, 6)  # Thinner margins
        
        # Add icon for system messages
        icon_label = QLabel()
        if self.is_error:
            icon_label.setText("⚠️")
        else:
            icon_label.setText("ℹ️")
        icon_label.setStyleSheet("font-size: 16px; margin-right: 8px;")
        container_layout.addWidget(icon_label)
        
        # Message text
        message_label = QLabel(self.message)
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        # Add to container
        container_layout.addWidget(message_label, 1)  # 1 = stretch factor
        
        # Add container to main layout with center alignment
        layout.setAlignment(Qt.AlignCenter)
        
        # Style based on error status and dark mode
        if self.is_error:
            if self.dark_mode:
                container.setStyleSheet(
                    "#messageContainer { "
                    "background-color: #4A1515; "
                    "border-radius: 12px; "
                    "margin-left: 60px; "
                    "margin-right: 60px; "
                    "border: 0.5px solid #5D2929; "
                    "} "
                    "QLabel { "
                    "color: #FFCDD2; "
                    "font-size: 13px; "
                    "}"
                )
            else:
                container.setStyleSheet(
                    "#messageContainer { "
                    "background-color: #FFEBEE; "
                    "border-radius: 12px; "
                    "margin-left: 60px; "
                    "margin-right: 60px; "
                    "border: 0.5px solid #FFCDD2; "
                    "} "
                    "QLabel { "
                    "color: #D32F2F; "
                    "font-size: 13px; "
                    "}"
                )
        else:
            if self.dark_mode:
                container.setStyleSheet(
                    "#messageContainer { "
                    "background-color: #1E3B1E; "
                    "border-radius: 12px; "
                    "margin-left: 60px; "
                    "margin-right: 60px; "
                    "border: 0.5px solid #2C4F2C; "
                    "} "
                    "QLabel { "
                    "color: #C8E6C9; "
                    "font-size: 13px; "
                    "}"
                )
            else:
                container.setStyleSheet(
                    "#messageContainer { "
                    "background-color: #E8F5E9; "
                    "border-radius: 12px; "
                    "margin-left: 60px; "
                    "margin-right: 60px; "
                    "border: 0.5px solid #C8E6C9; "
                    "} "
                    "QLabel { "
                    "color: #2E7D32; "
                    "font-size: 13px; "
                    "}"
                )
        
        layout.addWidget(container)

class ImageMessageBubble(QWidget):
    """Custom widget for image message bubbles"""
    
    def __init__(self, image_path, caption="", is_user=True, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.caption = caption
        self.is_user = is_user
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        # Check if dark mode is enabled
        self.dark_mode = self.is_dark_mode()
        self.init_ui()
        
    def is_dark_mode(self):
        """Check if the application is in dark mode by looking at the palette"""
        app = QApplication.instance()
        if app:
            palette = app.palette()
            bg_color = palette.color(QPalette.Window)
            # If background color is dark, we're in dark mode
            return bg_color.lightness() < 128
        return False
        
    def init_ui(self):
        """Initialize the UI"""
        # Main layout with more space for content
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Header with avatar and name
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 0, 12, 0)
        header_layout.setSpacing(8)
        
        # Create avatar label (circle with initials or icon)
        avatar = QLabel()
        avatar.setFixedSize(24, 24)
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setStyleSheet("font-weight: bold; color: white; border-radius: 12px;")
        
        # Create name label
        name_label = QLabel()
        
        # Set styles based on dark mode
        if self.dark_mode:
            user_color = "#7AB0FF"
            ai_color = "#E0E0E0"
        else:
            user_color = "#128C7E"
            ai_color = "#9370DB"
        
        if self.is_user:
            # User avatar - blue circle with "U"
            avatar.setText("U")
            avatar.setStyleSheet(f"background-color: {user_color}; color: white; border-radius: 12px; font-weight: bold;")
            name_label.setText("You")
            name_label.setStyleSheet(f"color: {user_color}; font-weight: bold; font-size: 13px;")
            header_layout.setAlignment(Qt.AlignRight)
            header_layout.addWidget(name_label)
            header_layout.addWidget(avatar)
        else:
            # AI avatar - purple circle with "AI"
            avatar.setText("AI")
            avatar.setStyleSheet(f"background-color: {ai_color}; color: white; border-radius: 12px; font-weight: bold; font-size: 10px;")
            name_label.setText("ChatMate Assistant")
            name_label.setStyleSheet(f"color: {ai_color}; font-weight: bold; font-size: 13px;")
            header_layout.addWidget(avatar)
            header_layout.addWidget(name_label)
            header_layout.addStretch(1)
        
        # Image container with rounded corners
        container = QFrame()
        container.setObjectName("imageContainer")
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        # Container layout
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(10, 6, 10, 6)  # Thinner margins
        
        # Image label
        image_label = QLabel()
        image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        # Load and scale image
        from PyQt5.QtGui import QPixmap
        pixmap = QPixmap(self.image_path)
        available_width = min(self.width() if self.width() > 0 else 400, 600)
        scaled_pixmap = pixmap.scaled(available_width, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_label.setPixmap(scaled_pixmap)
        
        # Add caption if provided
        if self.caption:
            caption_label = QLabel(self.caption)
            caption_label.setWordWrap(True)
            caption_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            caption_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            # Add caption to container
            container_layout.addWidget(image_label)
            container_layout.addWidget(caption_label)
        else:
            # Just add the image
            container_layout.addWidget(image_label)
        
        # Store the image label for resizing
        self.image_label = image_label
        self.pixmap = pixmap
        
        # Add header and container to main layout with appropriate alignment
        if self.is_user:
            layout.addWidget(header, 0, Qt.AlignRight)
            if self.dark_mode:
                container.setStyleSheet(
                    "#imageContainer { "
                    "background-color: #1A3C5E; "
                    "border-radius: 12px; "
                    "border-top-right-radius: 4px; "
                    "margin-left: 48px; "
                    "margin-right: 12px; "
                    "border: 0.5px solid #2C5F8E; "
                    "} "
                    "QLabel { "
                    "color: #FFFFFF; "
                    "font-size: 14px; "
                    "}"
                )
            else:
                container.setStyleSheet(
                    "#imageContainer { "
                    "background-color: #E7F7FF; "
                    "color: #1A1A1A; "
                    "border-radius: 12px; "
                    "border-top-right-radius: 4px; "
                    "margin-left: 48px; "
                    "margin-right: 12px; "
                    "border: 0.5px solid #D1E7FA; "
                    "} "
                    "QLabel { "
                    "color: #1A1A1A; "
                    "font-size: 14px; "
                    "}"
                )
        else:
            layout.addWidget(header, 0, Qt.AlignLeft)
            if self.dark_mode:
                container.setStyleSheet(
                    "#imageContainer { "
                    "background-color: #2A2A2A; "
                    "border-radius: 12px; "
                    "border-top-left-radius: 4px; "
                    "margin-right: 48px; "
                    "margin-left: 12px; "
                    "border: 0.5px solid #3A3A3A; "
                    "} "
                    "QLabel { "
                    "color: #FFFFFF; "
                    "font-size: 14px; "
                    "}"
                )
            else:
                container.setStyleSheet(
                    "#imageContainer { "
                    "background-color: #F9F9FA; "
                    "color: #1A1A1A; "
                    "border-radius: 12px; "
                    "border-top-left-radius: 4px; "
                    "margin-right: 48px; "
                    "margin-left: 12px; "
                    "border: 0.5px solid #E4E4E7; "
                    "} "
                    "QLabel { "
                    "color: #1A1A1A; "
                    "font-size: 14px; "
                    "}"
                )
        
        layout.addWidget(container)
            
    def resizeEvent(self, event):
        """Handle resize events to scale images appropriately"""
        if hasattr(self, 'image_label') and hasattr(self, 'pixmap'):
            # Get available width (accounting for margins and padding)
            available_width = min(self.width() - 80 if self.width() > 80 else 300, 600)
            
            # Scale image to fit available width while maintaining aspect ratio
            scaled_pixmap = self.pixmap.scaled(available_width, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
        
        # Call the parent class's resize event handler
        super().resizeEvent(event)

class DatabaseManager:
    """Class to manage SQLite database operations"""
    
    def __init__(self, db_file=DB_FILE):
        self.db_file = db_file
        self.conn = None
        self.init_db()
    
    def init_db(self):
        """Initialize the database and create tables if they don't exist"""
        try:
            self.conn = sqlite3.connect(self.db_file)
            cursor = self.conn.cursor()
            
            # Create todos table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS todos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            ''')
            
            # Create conversations table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                ai_mood TEXT DEFAULT 'professional'
            )
            ''')
            
            # Create messages table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                image_path TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
            ''')
            
            # Create settings table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT NOT NULL
            )
            ''')
            
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return False
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
    
    def get_todos(self):
        """Get all todos from database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, task, status, created_at FROM todos")
            rows = cursor.fetchall()
            
            todos = []
            for row in rows:
                todos.append({
                    "id": row[0],
                    "task": row[1],
                    "status": row[2],
                    "created_at": row[3]
                })
            
            return todos
        except sqlite3.Error as e:
            print(f"Error getting todos: {e}")
            return []
    
    def add_todo(self, task, status="To Do"):
        """Add a new todo to the database"""
        try:
            cursor = self.conn.cursor()
            created_at = datetime.now().isoformat()
            
            cursor.execute(
                "INSERT INTO todos (task, status, created_at) VALUES (?, ?, ?)",
                (task, status, created_at)
            )
            
            self.conn.commit()
            todo_id = cursor.lastrowid
            
            return {
                "id": todo_id,
                "task": task,
                "status": status,
                "created_at": created_at
            }
        except sqlite3.Error as e:
            print(f"Error adding todo: {e}")
            return None
    
    def update_todo(self, todo_id, status):
        """Update a todo's status"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE todos SET status = ? WHERE id = ?",
                (status, todo_id)
            )
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"Error updating todo: {e}")
            return False
    
    def delete_todo(self, todo_id):
        """Delete a todo"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"Error deleting todo: {e}")
            return False
    
    def get_conversations(self):
        """Get all conversations"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, title, created_at, updated_at, ai_mood FROM conversations ORDER BY updated_at DESC")
            rows = cursor.fetchall()
            
            conversations = []
            for row in rows:
                conversations.append({
                    "id": row[0],
                    "title": row[1],
                    "created_at": row[2],
                    "updated_at": row[3],
                    "ai_mood": row[4]
                })
            
            return conversations
        except sqlite3.Error as e:
            print(f"Error getting conversations: {e}")
            return []
    
    def create_conversation(self, title, ai_mood="professional"):
        """Create a new conversation"""
        try:
            cursor = self.conn.cursor()
            now = datetime.now().isoformat()
            
            cursor.execute(
                "INSERT INTO conversations (title, created_at, updated_at, ai_mood) VALUES (?, ?, ?, ?)",
                (title, now, now, ai_mood)
            )
            
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error creating conversation: {e}")
            return None
    
    def update_conversation(self, conversation_id, title=None, ai_mood=None):
        """Update a conversation"""
        try:
            cursor = self.conn.cursor()
            now = datetime.now().isoformat()
            updates = []
            params = []
            
            if title is not None:
                updates.append("title = ?")
                params.append(title)
            
            if ai_mood is not None:
                updates.append("ai_mood = ?")
                params.append(ai_mood)
            
            if updates:
                updates.append("updated_at = ?")
                params.append(now)
                params.append(conversation_id)
                
                query = f"UPDATE conversations SET {', '.join(updates)} WHERE id = ?"
                cursor.execute(query, params)
                self.conn.commit()
                
                return cursor.rowcount > 0
            return False
        except sqlite3.Error as e:
            print(f"Error updating conversation: {e}")
            return False
    
    def delete_conversation(self, conversation_id):
        """Delete a conversation and its messages"""
        try:
            cursor = self.conn.cursor()
            
            # Delete messages first (foreign key constraint)
            cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            
            # Delete the conversation
            cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"Error deleting conversation: {e}")
            return False
    
    def get_messages(self, conversation_id):
        """Get all messages for a conversation"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT id, role, content, timestamp, image_path FROM messages WHERE conversation_id = ? ORDER BY timestamp",
                (conversation_id,)
            )
            rows = cursor.fetchall()
            
            messages = []
            for row in rows:
                messages.append({
                    "id": row[0],
                    "role": row[1],
                    "content": row[2],
                    "timestamp": row[3],
                    "image_path": row[4]
                })
            
            return messages
        except sqlite3.Error as e:
            print(f"Error getting messages: {e}")
            return []
    
    def add_message(self, conversation_id, role, content, image_path=None):
        """Add a message to a conversation"""
        try:
            cursor = self.conn.cursor()
            timestamp = datetime.now().isoformat()
            
            cursor.execute(
                "INSERT INTO messages (conversation_id, role, content, timestamp, image_path) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, role, content, timestamp, image_path)
            )
            
            # Update conversation's updated_at timestamp
            cursor.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (timestamp, conversation_id)
            )
            
            self.conn.commit()
            
            return {
                "id": cursor.lastrowid,
                "role": role,
                "content": content,
                "timestamp": timestamp,
                "image_path": image_path
            }
        except sqlite3.Error as e:
            print(f"Error adding message: {e}")
            return None
    
    def get_setting(self, key, default=None):
        """Get a setting value"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            
            if row:
                return row[0]
            return default
        except sqlite3.Error as e:
            print(f"Error getting setting: {e}")
            return default
    
    def set_setting(self, key, value):
        """Set a setting value"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                (key, value)
            )
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error setting setting: {e}")
            return False


class ChatHistory:
    """Class to store and manage chat history (legacy JSON-based version)"""
    
    def __init__(self, max_history=10):
        self.max_history = max_history
        self.history = []
        self.history_file = "chat_history.json"
    
    def add_message(self, role, content, image_path=None):
        """Add a message to the history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path
        }
        
        self.history.append(message)
        
        # Limit history size
        if len(self.history) > self.max_history * 100:  # Allow ~100 messages per conversation
            self.history = self.history[-self.max_history * 100:]
            
        # Save to file
        self.save_history()
        
        return message
    
    def get_history(self):
        return self.history
    
    def clear_history(self):
        self.history = []
        self.save_history()
    
    def save_history(self):
        try:
            with open(self.history_file, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Error saving chat history: {e}")
    
    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    self.history = json.load(f)
            except Exception as e:
                print(f"Error loading chat history: {e}")
                self.history = []


class ChatMateApp(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.db = DatabaseManager()
        self.todos = []
        self.chatbot = ChatbotAPI()
        self.chat_history_manager = ChatHistory(max_history=10)  # Legacy system
        self.current_theme = "dark"  # Default theme
        self.current_conversation_id = None
        self.conversations = []
        self.init_ui()
        self.load_todos()
        self.load_conversations()
        self.apply_theme(self.current_theme)
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("ChatMate")
        self.setMinimumSize(600, 450)  # Smaller minimum size for better compatibility
        self.resize(900, 700)  # Default size that works well
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)  # Smaller margins for more content space
        main_layout.setSpacing(5)  # Tighter spacing
        
        # Create tab widget with responsive sizing policy
        self.tab_widget = QTabWidget()
        self.tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tab_widget.setDocumentMode(True)  # More modern look
        self.tab_widget.setMovable(True)  # Allow tab reordering
        
        # Create todo tab with responsive layout
        todo_widget = QWidget()
        todo_layout = QVBoxLayout(todo_widget)
        todo_layout.setContentsMargins(8, 8, 8, 8)
        todo_layout.setSpacing(8)
        
        # App title
        title_label = QLabel("MY TASKS")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 24px; 
            font-weight: bold; 
            margin: 10px; 
            color: #d7ba7d; 
            font-family: 'Consolas', 'Monaco', monospace;
            border-bottom: 1px solid #3c3c3c;
            padding-bottom: 10px;
        """)
        
        # Add new todo section
        input_layout = QHBoxLayout()
        self.task_input = QLineEdit()
        self.task_input.setPlaceholderText("Add a new task...")
        self.task_input.returnPressed.connect(self.add_todo)
        
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self.add_todo)
        
        input_layout.addWidget(self.task_input)
        input_layout.addWidget(add_btn)
        
        # Category tabs
        self.category_tabs = QTabWidget()
        
        # Create a tab for each category with responsive sizing
        self.category_lists = {}
        for category in CATEGORIES:
            list_widget = QListWidget()
            list_widget.setSpacing(5)
            list_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            list_widget.setVerticalScrollMode(QListWidget.ScrollPerPixel)  # Smooth scrolling
            self.category_lists[category] = list_widget
            self.category_tabs.addTab(list_widget, f"{category} (0)")
        
        # Add widgets to todo layout
        todo_layout.addWidget(title_label)
        todo_layout.addLayout(input_layout)
        todo_layout.addWidget(self.category_tabs)
        
        # Create chat tab with responsive layout
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        chat_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for chat area
        chat_layout.setSpacing(0)  # No spacing for chat components
        
        # AI model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("AI Model:")
        model_label.setStyleSheet("color: #d7ba7d; font-weight: bold;")
        
        self.model_selector = QComboBox()
        self.model_selector.addItems(self.chatbot.models if self.chatbot.models else ["No models available"])
        if self.chatbot.model:
            self.model_selector.setCurrentText(self.chatbot.model)
        self.model_selector.currentTextChanged.connect(self.change_ai_model)
        
        model_status = QLabel()
        if self.chatbot.offline_mode:
            model_status.setText("⚠️ Offline")
            model_status.setStyleSheet("color: #ff6b6b;")
        else:
            model_status.setText("✓ Online")
            model_status.setStyleSheet("color: #4ec9b0;")
        self.model_status_label = model_status
        
        refresh_btn = QPushButton("⟳")
        refresh_btn.setToolTip("Refresh model list")
        refresh_btn.setFixedWidth(30)
        refresh_btn.clicked.connect(self.refresh_models)
        
        # Online API configuration button
        online_api_btn = QPushButton("☁️")
        online_api_btn.setToolTip("Configure Online AI API")
        online_api_btn.setFixedWidth(30)
        online_api_btn.clicked.connect(self.configure_online_api)
        
        # Online API indicator
        self.online_api_label = QLabel()
        if self.chatbot.use_online_api and self.chatbot.online_api.available:
            self.online_api_label.setText(f"☁️ {self.chatbot.online_api.provider.capitalize()}")
            self.online_api_label.setStyleSheet("color: #4ec9b0;")
        else:
            self.online_api_label.setText("☁️ Off")
            self.online_api_label.setStyleSheet("color: #6c6c6c;")
        
        # AI Mood selector
        mood_label = QLabel("AI Mood:")
        mood_label.setStyleSheet("color: #d7ba7d; font-weight: bold;")
        
        self.mood_selector = QComboBox()
        for mood_key, mood_data in AI_MOODS.items():
            self.mood_selector.addItem(mood_data["name"], mood_key)
        
        # Set current mood
        current_mood_index = self.mood_selector.findData(self.chatbot.current_mood)
        if current_mood_index >= 0:
            self.mood_selector.setCurrentIndex(current_mood_index)
        
        self.mood_selector.currentIndexChanged.connect(self.change_ai_mood)
        
        # Mood indicator
        self.mood_indicator = QLabel()
        self.update_mood_indicator()
        
        clear_history_btn = QPushButton("Clear History")
        clear_history_btn.clicked.connect(self.clear_chat_history)
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_selector, 1)  # 1 = stretch factor
        model_layout.addWidget(model_status)
        model_layout.addWidget(refresh_btn)
        model_layout.addWidget(online_api_btn)
        model_layout.addWidget(self.online_api_label)
        model_layout.addWidget(mood_label)
        model_layout.addWidget(self.mood_selector)
        model_layout.addWidget(self.mood_indicator)
        model_layout.addWidget(clear_history_btn)
        
        # Chat history display - using a scroll area with widgets instead of QTextEdit
        chat_scroll = QScrollArea()
        chat_scroll.setWidgetResizable(True)
        chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        chat_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        chat_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Chat container for chat messages with responsive layout
        self.chat_container = QWidget()
        self.chat_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setSpacing(12)  # Slightly more space between messages
        self.chat_layout.setContentsMargins(10, 15, 10, 15)  # More padding for better readability
        
        # Set the container as the scroll area widget
        chat_scroll.setWidget(self.chat_container)
        
        # Set WhatsApp-like background
        chat_scroll.setStyleSheet("""
            QScrollArea {
                background-color: #0D1418; /* WhatsApp dark background */
                border: none;
            }
            QScrollBar:vertical {
                background: #1A2C37;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #374248;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # Load previous chat history
        self.display_chat_history()
        
        # Theme selector
        theme_layout = QHBoxLayout()
        theme_label = QLabel("Theme:")
        theme_label.setStyleSheet("color: #AAAAAA; font-size: 12px;")
        
        self.theme_selector = QComboBox()
        self.theme_selector.addItems(["Dark", "Light", "Blue"])
        self.theme_selector.setCurrentText("Dark")
        self.theme_selector.currentTextChanged.connect(self.change_theme)
        self.theme_selector.setFixedWidth(100)
        
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_selector)
        theme_layout.addStretch(1)
        
        # Chat input area with WhatsApp-like styling
        chat_input_container = QWidget()
        chat_input_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        chat_input_container.setStyleSheet("""
            QWidget {
                background-color: #1E2428; /* WhatsApp input area */
                border-top: 1px solid #2D383E;
            }
        """)
        chat_input_layout = QHBoxLayout(chat_input_container)
        chat_input_layout.setContentsMargins(10, 5, 10, 10)
        chat_input_layout.setSpacing(8)  # Consistent spacing
        
        # Image upload button
        upload_btn = QPushButton()
        upload_btn.setIcon(QIcon.fromTheme("image"))
        upload_btn.setMinimumSize(36, 36)
        upload_btn.setMaximumSize(40, 40)  # Allow slight growth on larger screens
        upload_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        upload_btn.setToolTip("Upload an image")
        upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #00A884; /* WhatsApp accent green */
                border-radius: 20px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #128C7E;
            }
        """)
        upload_btn.clicked.connect(self.upload_image)
        
        # Online API settings button
        cloud_btn = QPushButton()
        cloud_btn.setIcon(QIcon.fromTheme("cloud"))
        cloud_btn.setMinimumSize(36, 36)
        cloud_btn.setMaximumSize(40, 40)  # Allow slight growth on larger screens
        cloud_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        cloud_btn.setToolTip("Configure online AI services")
        cloud_btn.setStyleSheet("""
            QPushButton {
                background-color: #00A884; /* WhatsApp accent green */
                border-radius: 20px;
                border: none;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #009673;
            }
            QPushButton:pressed {
                background-color: #008C6A;
            }
        """)
        cloud_btn.clicked.connect(self.show_online_api_dialog)
        
        # Chat input field with responsive sizing
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type a message...")
        self.chat_input.returnPressed.connect(self.send_message)
        self.chat_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.chat_input.setMinimumHeight(40)
        self.chat_input.setStyleSheet("""
            QLineEdit {
                background-color: #2A3942; /* WhatsApp input field */
                color: #D1D7DB;
                border-radius: 18px;
                padding: 10px 15px;
                font-size: 14px;
                border: none;
            }
        """)
        
        # Send button
        self.send_btn = QPushButton()  # Make it an instance variable so we can change it
        self.send_btn.setIcon(QIcon.fromTheme("document-send"))
        self.send_btn.setMinimumSize(36, 36)
        self.send_btn.setMaximumSize(40, 40)  # Allow slight growth on larger screens
        self.send_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.send_btn.clicked.connect(self.send_message)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #00A884; /* WhatsApp accent green */
                border-radius: 20px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #128C7E;
            }
        """)
        
        # Audio input button
        audio_btn = QPushButton()
        audio_btn.setIcon(QIcon.fromTheme("audio-input-microphone"))
        audio_btn.setMinimumSize(36, 36)
        audio_btn.setMaximumSize(40, 40)
        audio_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        audio_btn.setToolTip("Voice input")
        audio_btn.setStyleSheet("""
            QPushButton {
                background-color: #00A884; /* WhatsApp accent green */
                border-radius: 20px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #128C7E;
            }
        """)
        audio_btn.clicked.connect(self.record_audio)
        
        # Image generation button (only visible for capable models)
        self.generate_image_btn = QPushButton()
        self.generate_image_btn.setIcon(QIcon.fromTheme("insert-image"))
        self.generate_image_btn.setMinimumSize(36, 36)
        self.generate_image_btn.setMaximumSize(40, 40)
        self.generate_image_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.generate_image_btn.setToolTip("Generate an image with AI")
        self.generate_image_btn.setStyleSheet("""
            QPushButton {
                background-color: #9370DB; /* Purple for image generation */
                border-radius: 20px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #8A2BE2;
            }
        """)
        self.generate_image_btn.clicked.connect(self.show_image_generation_dialog)
        self.generate_image_btn.setVisible(self.chatbot.can_generate_images())
        
        # Add buttons to layout
        chat_input_layout.addWidget(upload_btn)
        chat_input_layout.addWidget(cloud_btn)
        chat_input_layout.addWidget(audio_btn)
        chat_input_layout.addWidget(self.generate_image_btn)  # Add image generation button
        chat_input_layout.addWidget(self.chat_input)
        chat_input_layout.addWidget(self.send_btn)
        
        # Add all widgets to chat layout
        chat_title = QLabel("ChatMate Assistant")
        chat_title.setStyleSheet("""
            font-size: 18px; 
            font-weight: bold; 
            color: #E9EDF0;
            padding: 10px;
        """)
        chat_title.setAlignment(Qt.AlignCenter)
        
        chat_header = QWidget()
        chat_header_layout = QHBoxLayout(chat_header)
        chat_header_layout.setContentsMargins(10, 5, 10, 5)
        
        chat_title = QLabel("ChatMate Assistant")
        chat_title.setStyleSheet("""
            font-size: 18px; 
            font-weight: bold; 
            color: #E9EDF0;
            padding: 5px;
        """)
        
        # Conversation selector
        self.conversation_selector = QComboBox()
        self.conversation_selector.setMinimumWidth(200)
        self.conversation_selector.setStyleSheet("""
            background-color: #2A3942;
            color: #D1D7DB;
            border-radius: 4px;
            padding: 5px;
            font-size: 14px;
        """)
        self.conversation_selector.currentIndexChanged.connect(self.change_conversation)
        
        # New conversation button
        new_chat_btn = QPushButton("+")
        new_chat_btn.setToolTip("New Conversation")
        new_chat_btn.setFixedSize(30, 30)
        new_chat_btn.setStyleSheet("""
            QPushButton {
                background-color: #00A884;
                color: white;
                border-radius: 15px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #128C7E;
            }
        """)
        new_chat_btn.clicked.connect(self.new_conversation)
        
        # Delete conversation button
        delete_chat_btn = QPushButton("×")
        delete_chat_btn.setToolTip("Delete Current Conversation")
        delete_chat_btn.setFixedSize(30, 30)
        delete_chat_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF5252;
                color: white;
                border-radius: 15px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #FF0000;
            }
        """)
        delete_chat_btn.clicked.connect(self.delete_current_conversation)
        
        # Settings button
        settings_btn = QPushButton()
        settings_btn.setIcon(QIcon.fromTheme("preferences-system"))
        settings_btn.setToolTip("Model Settings & Local Model Scan")
        settings_btn.setFixedSize(30, 30)
        settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
                border-radius: 15px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #455A64;
            }
        """)
        settings_btn.clicked.connect(self.show_model_settings_dialog)
        
        chat_header_layout.addWidget(chat_title)
        chat_header_layout.addStretch(1)
        chat_header_layout.addWidget(self.conversation_selector)
        chat_header_layout.addWidget(new_chat_btn)
        chat_header_layout.addWidget(delete_chat_btn)
        chat_header_layout.addWidget(settings_btn)
        
        chat_layout.addWidget(chat_header)
        chat_layout.addLayout(theme_layout)  # Add theme selector
        chat_layout.addWidget(chat_scroll)
        chat_layout.addWidget(chat_input_container)
        
        # Add tabs to tab widget
        self.tab_widget.addTab(todo_widget, "Tasks")
        self.tab_widget.addTab(chat_widget, "ChatMate")
        
        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)
        
        # Set the central widget
        self.setCentralWidget(main_widget)
        
        # Apply styles
        self.apply_styles()
        
    def apply_styles(self):
        """Apply styles to the application - code editor theme"""
        # Default dark theme styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QTabWidget::pane {
                border: 1px solid #3c3c3c;
                background-color: #252526;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #adadad;
                border: 1px solid #3c3c3c;
                border-bottom-color: #3c3c3c;
                min-width: 8ex;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #1e1e1e;
                color: #ffffff;
                border-bottom-color: #007acc;
                border-bottom-width: 2px;
            }
            QTabBar::tab:hover:!selected {
                background-color: #323232;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #3c3c3c;
                border-radius: 2px;
                background-color: #252526;
                color: #d4d4d4;
                font-family: 'Consolas', 'Monaco', monospace;
            }
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                border-radius: 2px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #0d5ca0;
            }
            QListWidget {
                border: 1px solid #3c3c3c;
                border-radius: 2px;
                background-color: #252526;
                color: #d4d4d4;
                padding: 5px;
                font-family: 'Consolas', 'Monaco', monospace;
                alternate-background-color: #2a2a2a;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #3c3c3c;
            }
            QListWidget::item:selected {
                background-color: #094771;
                color: #ffffff;
            }
            QListWidget::item:hover:!selected {
                background-color: #2a2d2e;
            }
            QTextEdit {
                border: 1px solid #3c3c3c;
                border-radius: 2px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                padding: 5px;
                font-family: 'Consolas', 'Monaco', monospace;
                selection-background-color: #264f78;
            }
            QScrollBar:vertical {
                border: none;
                background: #1e1e1e;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #424242;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #4f4f4f;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                border: none;
                background: #1e1e1e;
                height: 10px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: #424242;
                min-width: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #4f4f4f;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QComboBox {
                border: 1px solid #3c3c3c;
                border-radius: 2px;
                padding: 5px 10px;
                background-color: #252526;
                color: #d4d4d4;
                min-height: 20px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #3c3c3c;
                background-color: #252526;
            }
            QComboBox QAbstractItemView {
                background-color: #252526;
                color: #d4d4d4;
                selection-background-color: #094771;
                selection-color: #ffffff;
                border: 1px solid #3c3c3c;
            }
            QLabel {
                color: #d4d4d4;
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }
        """)
        
    def add_todo(self):
        """Add a new todo item"""
        task = self.task_input.text().strip()
        if not task:
            return
            
        # Create new todo
        todo = {
            "id": len(self.todos) + 1,
            "task": task,
            "status": "To Do",
            "created_at": datetime.now().isoformat()
        }
        
        self.todos.append(todo)
        self.add_todo_widget(todo)
        self.save_todos()
        
        # Clear input
        self.task_input.clear()
        
    def add_todo_widget(self, todo):
        """Add a todo widget to the appropriate list"""
        item = QListWidgetItem()
        todo_widget = TodoItem(todo)
        
        # Connect signals
        todo_widget.deleted.connect(self.delete_todo)
        todo_widget.status_changed.connect(self.update_todo_status)
        
        # Get the list for this category
        list_widget = self.category_lists[todo["status"]]
        
        # Add to list
        list_widget.addItem(item)
        item.setSizeHint(todo_widget.sizeHint())
        list_widget.setItemWidget(item, todo_widget)
        
        # Update tab counts
        self.update_tab_counts()
        
    def delete_todo(self, todo):
        """Delete a todo item"""
        # Remove from data
        self.todos = [t for t in self.todos if t["id"] != todo["id"]]
        
        # Remove from UI
        list_widget = self.category_lists[todo["status"]]
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            widget = list_widget.itemWidget(item)
            if widget.todo["id"] == todo["id"]:
                list_widget.takeItem(i)
                break
                
        # Update tab counts and save
        self.update_tab_counts()
        self.save_todos()
        
    def update_todo_status(self, todo, new_status):
        """Update the status of a todo item"""
        # Find the todo in our list
        for t in self.todos:
            if t["id"] == todo["id"]:
                t["status"] = new_status
                break
                
        # Remove from old list
        old_list = self.category_lists[todo["status"]]
        for i in range(old_list.count()):
            item = old_list.item(i)
            widget = old_list.itemWidget(item)
            if widget.todo["id"] == todo["id"]:
                old_list.takeItem(i)
                break
                
        # Add to new list
        self.add_todo_widget(todo)
        
        # Update tab counts and save
        self.update_tab_counts()
        self.save_todos()
        
    def update_tab_counts(self):
        """Update the category tab counts"""
        for i, category in enumerate(CATEGORIES):
            count = self.category_lists[category].count()
            self.category_tabs.setTabText(i, f"{category} ({count})")
            
    def change_ai_model(self, model_name):
        """Change the AI model"""
        if model_name and model_name != "No models available":
            success = self.chatbot.set_model(model_name)
            if success:
                # Update UI based on model capabilities
                self.update_ui_for_model_capabilities()
                
                # Add system message about model change
                self.add_system_message(f"Switched to model: {model_name}")
                
                # Check if model supports image generation
                if self.chatbot.can_generate_images():
                    self.add_system_message(f"This model supports image generation!")
            else:
                self.add_system_message(f"Failed to switch to model: {model_name}", is_error=True)
    
    def update_ui_for_model_capabilities(self):
        """Update UI elements based on current model capabilities"""
        # Update image generation button visibility
        if hasattr(self, 'generate_image_btn'):
            self.generate_image_btn.setVisible(self.chatbot.can_generate_images())
            self.generate_image_btn.setEnabled(self.chatbot.can_generate_images())
    
    def configure_online_api(self):
        """Configure online API settings"""
        dialog = OnlineAPIDialog(self.chatbot.online_api)
        if dialog.exec_():
            # Update API key and provider
            self.chatbot.online_api.api_key = dialog.api_key
            self.chatbot.online_api.provider = dialog.provider
            self.chatbot.online_api.available = True
            self.add_system_message(f"Connected to {dialog.provider} API")
            
    def show_online_api_dialog(self):
        """Show dialog to configure online API settings"""
        # This is an alias for configure_online_api for better readability
        self.configure_online_api()
        
    def update_mood_indicator(self):
        """Update the mood indicator label"""
        if hasattr(self, 'mood_indicator') and hasattr(self.chatbot, 'current_mood'):
            mood = self.chatbot.current_mood
            if mood in AI_MOODS:
                color = AI_MOODS[mood]["color"]
                self.mood_indicator.setText("●")
                self.mood_indicator.setStyleSheet(f"color: {color}; font-size: 16px;")
                self.mood_indicator.setToolTip(AI_MOODS[mood]["description"])
    
    def change_ai_mood(self, index):
        """Change the AI assistant's mood"""
        mood_key = self.mood_selector.itemData(index)
        if mood_key and mood_key in AI_MOODS:
            self.chatbot.current_mood = mood_key
            self.update_mood_indicator()
            
            # Update current conversation mood if one exists
            if self.current_conversation_id:
                self.db.update_conversation(self.current_conversation_id, ai_mood=mood_key)
            
            # Add system message about mood change
            mood_name = AI_MOODS[mood_key]["name"]
            self.add_system_message(f"AI mood changed to: {mood_name}")
    
    def change_theme(self, theme_name):
        """Change the application theme"""
        theme_name = theme_name.lower()
        if theme_name in THEMES:
            self.current_theme = theme_name
            self.apply_theme(theme_name)
            self.add_system_message(f"Theme changed to {theme_name.capitalize()}")
        
    def apply_theme(self, theme_name):
        """Apply the selected theme to the application"""
        if theme_name not in THEMES:
            return
            
        theme = THEMES[theme_name]
        
        # Apply theme to main window
        self.setStyleSheet(f"""
            QMainWindow, QDialog {{
                background-color: {theme['app_bg']};
                color: {theme['text']};
            }}
            QWidget {{
                background-color: {theme['app_bg']};
                color: {theme['text']};
            }}
            QTabWidget::pane {{
                border: 1px solid {theme['border']};
                background-color: {theme['sidebar_bg']};
            }}
            QTabBar::tab {{
                background-color: {theme['sidebar_bg']};
                color: {theme['secondary_text']};
                border: 1px solid {theme['border']};
                padding: 6px 12px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background-color: {theme['app_bg']};
                color: {theme['text']};
                border-bottom-color: {theme['app_bg']};
            }}
            QLineEdit {{
                background-color: {theme['input_bg']};
                color: {theme['text']};
                border: 1px solid {theme['border']};
                padding: 5px;
            }}
            QComboBox {{
                background-color: {theme['input_bg']};
                color: {theme['text']};
                border: 1px solid {theme['border']};
                padding: 5px;
            }}
            QScrollArea {{
                background-color: {theme['chat_bg']};
                border: none;
            }}
        """)
        
        # Update chat container background
        if hasattr(self, 'chat_container'):
            self.chat_container.setStyleSheet(f"background-color: {theme['chat_bg']};")
        
        # Update theme selector
        if hasattr(self, 'theme_selector'):
            self.theme_selector.setCurrentText(theme_name.capitalize())
    
    def set_processing_state(self, is_processing):
        """Change the send button appearance based on processing state"""
        if is_processing:
            # Change to square stop button
            self.send_btn.setIcon(QIcon.fromTheme("process-stop"))
            self.send_btn.setToolTip("Stop generation")
            self.send_btn.setStyleSheet("""
                QPushButton {
                    background-color: #E74C3C; /* Red for stop */
                    border-radius: 20px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #C0392B;
                }
            """)
            # Force update
            QApplication.processEvents()
        else:
            # Change back to send arrow
            self.send_btn.setIcon(QIcon.fromTheme("document-send"))
            self.send_btn.setToolTip("Send message")
            self.send_btn.setStyleSheet("""
                QPushButton {
                    background-color: #00A884; /* WhatsApp accent green */
                    border-radius: 20px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #128C7E;
                }
            """)
            # Force update
            QApplication.processEvents()
    
    def record_audio(self):
        """Record audio input from the user"""
        try:
            # Show a recording indicator
            self.add_system_message("Recording audio... (speak now)")
            
            # Simulate audio recording (in a real app, we would use PyAudio or similar)
            # For now, just wait a moment and then add a message
            QApplication.processEvents()
            
            # In a real implementation, we would record audio and transcribe it
            # Since we don't have audio libraries installed, we'll simulate it
            self.chat_input.setText("This is simulated voice input from audio recording")
            
            # Add a confirmation message
            self.add_system_message("Audio recording completed")
            
        except Exception as e:
            self.add_system_message(f"Error recording audio: {str(e)}", is_error=True)
    
    def refresh_models(self):
        """Refresh the list of available models"""
        self.chat_history.append(f"<span style='color:#4ec9b0;'><b>System:</b> Refreshing available models...</span>")
        
        # Get available models
        self.chatbot.get_available_models()
        
        # Update model selector
        current_model = self.model_selector.currentText()
        self.model_selector.clear()
        self.model_selector.addItems(self.chatbot.models if self.chatbot.models else ["No models available"])
        
        # Try to restore previous selection
        if current_model in self.chatbot.models:
            self.model_selector.setCurrentText(current_model)
        elif self.chatbot.model:
            self.model_selector.setCurrentText(self.chatbot.model)
            
        # Update status label
        if self.chatbot.offline_mode:
            self.model_status_label.setText("⚠️ Offline")
            self.model_status_label.setStyleSheet("color: #ff6b6b;")
            self.chat_history.append(f"<span style='color:#ff6b6b;'><b>System:</b> Ollama service is offline.</span>")
        else:
            self.model_status_label.setText("✓ Online")
            self.model_status_label.setStyleSheet("color: #4ec9b0;")
            self.chat_history.append(f"<span style='color:#4ec9b0;'><b>System:</b> Found {len(self.chatbot.models)} models. Using {self.chatbot.model}.</span>")
            
        self.chat_history.append("")
        
        # Scroll to bottom
        self.chat_history.verticalScrollBar().setValue(
            self.chat_history.verticalScrollBar().maximum()
        )
    
    def upload_image(self):
        """Upload an image for analysis"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                image_path = file_paths[0]
                self.add_image_message(image_path)
                self.analyze_image(image_path)
                
    def show_image_generation_dialog(self):
        """Show dialog to generate an image with AI"""
        # Create a custom dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Generate Image with AI")
        dialog.setMinimumWidth(400)
        
        # Layout
        layout = QVBoxLayout(dialog)
        
        # Prompt input
        prompt_label = QLabel("Describe the image you want to generate:")
        prompt_input = QTextEdit()
        prompt_input.setPlaceholderText("E.g., A serene mountain landscape with a lake at sunset, photorealistic style")
        prompt_input.setMinimumHeight(100)
        
        # Size selection
        size_label = QLabel("Image size:")
        size_combo = QComboBox()
        size_combo.addItems(["256x256", "512x512", "1024x1024"])
        size_combo.setCurrentText("512x512")
        
        # Model info
        model_label = QLabel(f"Using model: {self.chatbot.model}")
        model_label.setStyleSheet("color: #888888; font-style: italic;")
        
        # Buttons
        button_box = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        generate_btn = QPushButton("Generate")
        generate_btn.setObjectName("generate_btn")  # Set object name for later reference
        generate_btn.setStyleSheet("background-color: #9370DB; color: white; font-weight: bold;")
        
        button_box.addWidget(cancel_btn)
        button_box.addWidget(generate_btn)
        
        # Add widgets to layout
        layout.addWidget(prompt_label)
        layout.addWidget(prompt_input)
        layout.addWidget(size_label)
        layout.addWidget(size_combo)
        layout.addWidget(model_label)
        layout.addLayout(button_box)
        
        # Progress indicator
        progress_label = QLabel("Generating image...")
        progress_label.setObjectName("progress_label")  # Set object name for later reference
        progress_label.setVisible(False)
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 0)  # Indeterminate progress
        progress_bar.setVisible(False)
        
        layout.addWidget(progress_label)
        layout.addWidget(progress_bar)
        
        # Connect signals
        cancel_btn.clicked.connect(dialog.reject)
        
        # Generation function
        def start_generation():
            prompt = prompt_input.toPlainText().strip()
            if not prompt:
                QMessageBox.warning(dialog, "Empty Prompt", "Please enter a description for the image.")
                return
                
            # Show progress
            generate_btn.setEnabled(False)
            prompt_input.setEnabled(False)
            size_combo.setEnabled(False)
            progress_label.setVisible(True)
            progress_bar.setVisible(True)
            
            # Use a timer to allow UI to update
            QTimer.singleShot(100, lambda: self.generate_ai_image(prompt, size_combo.currentText(), dialog))
            
        generate_btn.clicked.connect(start_generation)
        
        # Show dialog
        dialog.exec_()
        
    def show_model_settings_dialog(self):
        """Show dialog for model settings and local model scanning"""
        # Create a custom dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("AI Model Settings")
        dialog.setMinimumWidth(500)
        dialog.setMinimumHeight(400)
        
        # Main layout
        layout = QVBoxLayout(dialog)
        
        # Create tabs
        tabs = QTabWidget()
        model_tab = QWidget()
        scan_tab = QWidget()
        
        # Model selection tab
        model_layout = QVBoxLayout(model_tab)
        
        # Current model info
        current_model_frame = QFrame()
        current_model_frame.setFrameShape(QFrame.StyledPanel)
        current_model_frame.setStyleSheet("background-color: #2A3942; border-radius: 8px; padding: 10px;")
        current_model_layout = QVBoxLayout(current_model_frame)
        
        current_model_label = QLabel(f"Current Model: <b>{self.chatbot.model}</b>")
        current_model_label.setStyleSheet("color: #E9EDF0; font-size: 16px;")
        
        # Check if model supports image generation
        supports_images = self.chatbot.can_generate_images()
        capabilities_label = QLabel(f"Image Generation: {'✓ Supported' if supports_images else '✗ Not Supported'}")
        capabilities_label.setStyleSheet(f"color: {'#4CAF50' if supports_images else '#F44336'}; font-size: 14px;")
        
        current_model_layout.addWidget(current_model_label)
        current_model_layout.addWidget(capabilities_label)
        
        # Available models list
        models_label = QLabel("Available Models:")
        models_label.setStyleSheet("color: #E9EDF0; font-size: 14px; margin-top: 10px;")
        
        models_list = QListWidget()
        models_list.setStyleSheet("""
            QListWidget {
                background-color: #1E2428;
                color: #D1D7DB;
                border-radius: 4px;
                padding: 5px;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #2D383E;
            }
            QListWidget::item:selected {
                background-color: #00A884;
                color: white;
            }
        """)
        
        # Populate models list
        for model in self.chatbot.models:
            item = QListWidgetItem(model)
            # Add icon for image-capable models
            tooltip = ""
            if model in self.chatbot.model_capabilities:
                capabilities = self.chatbot.model_capabilities[model]
                
                # Set icon based on source
                source = capabilities.get("source", "")
                if source == "ollama_api" or source == "ollama_cli":
                    item.setIcon(QIcon.fromTheme("network-server"))
                    tooltip = "Ollama model"
                elif source == "file_scan":
                    item.setIcon(QIcon.fromTheme("drive-harddisk"))
                    tooltip = "Local model file"
                
                # Add image capability to tooltip
                if capabilities.get("image_generation", False):
                    if tooltip:
                        tooltip += " - "
                    tooltip += "Supports image generation"
                    
                # Add model size to tooltip if available
                if "size_mb" in capabilities and capabilities["size_mb"] > 0:
                    size_mb = capabilities["size_mb"]
                    if size_mb >= 1000:
                        size_gb = size_mb / 1024
                        if tooltip:
                            tooltip += " - "
                        tooltip += f"Size: {size_gb:.1f} GB"
                    else:
                        if tooltip:
                            tooltip += " - "
                        tooltip += f"Size: {size_mb:.0f} MB"
            
            if tooltip:
                item.setToolTip(tooltip)
                
            # Set current item
            if model == self.chatbot.model:
                models_list.setCurrentItem(item)
                
            models_list.addItem(item)
        
        # Button to switch model
        switch_model_btn = QPushButton("Switch to Selected Model")
        switch_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #00A884;
                color: white;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #009673;
            }
        """)
        
        # Check capabilities button
        check_capabilities_btn = QPushButton("Check Model Capabilities")
        check_capabilities_btn.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #455A64;
            }
        """)
        
        # Connect switch model button
        def switch_model():
            selected_items = models_list.selectedItems()
            if selected_items:
                model_name = selected_items[0].text()
                self.change_ai_model(model_name)
                # Update current model info
                current_model_label.setText(f"Current Model: <b>{self.chatbot.model}</b>")
                supports_images = self.chatbot.can_generate_images()
                capabilities_label.setText(f"Image Generation: {'✓ Supported' if supports_images else '✗ Not Supported'}")
                capabilities_label.setStyleSheet(f"color: {'#4CAF50' if supports_images else '#F44336'}; font-size: 14px;")
        
        # Connect check capabilities button
        def check_model_capabilities():
            selected_items = models_list.selectedItems()
            if not selected_items:
                QMessageBox.warning(dialog, "No Model Selected", "Please select a model to check capabilities.")
                return
                
            model_name = selected_items[0].text()
            
            # Create a progress dialog
            progress_dialog = QDialog(dialog)
            progress_dialog.setWindowTitle(f"Checking Capabilities: {model_name}")
            progress_dialog.setMinimumWidth(400)
            progress_dialog.setMinimumHeight(150)
            
            progress_layout = QVBoxLayout(progress_dialog)
            progress_label = QLabel(f"Checking capabilities of {model_name}...")
            progress_label.setStyleSheet("color: #E9EDF0; font-size: 14px;")
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 0)  # Indeterminate progress
            progress_bar.setStyleSheet("""
                QProgressBar {
                    background-color: #1E2428;
                    color: white;
                    border-radius: 4px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #00A884;
                    border-radius: 4px;
                }
            """)
            
            progress_layout.addWidget(progress_label)
            progress_layout.addWidget(progress_bar)
            
            # Show progress dialog
            progress_dialog.show()
            
            # Function to check capabilities in a separate thread
            def perform_check():
                try:
                    # Check capabilities with a timeout
                    result = self.chatbot.check_model_capabilities(model_name)
                    
                    # Close progress dialog if it's still open
                    try:
                        if progress_dialog and progress_dialog.isVisible():
                            progress_dialog.accept()
                    except Exception as dialog_error:
                        print(f"Error closing progress dialog: {str(dialog_error)}")
                    
                    # Always show results dialog, even if there was an issue
                    capabilities = result.get("capabilities", {"name": model_name})
                    
                    # Ensure we have at least the basic fields
                    if "name" not in capabilities:
                        capabilities["name"] = model_name
                    if "check_time" not in capabilities:
                        capabilities["check_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                        
                    # Show the capabilities dialog
                    try:
                        show_capabilities_dialog(model_name, capabilities)
                    except Exception as dialog_error:
                        print(f"Error showing capabilities dialog: {str(dialog_error)}")
                        QMessageBox.warning(
                            dialog,
                            "Error Showing Capabilities",
                            f"Could not display capabilities dialog: {str(dialog_error)}"
                        )
                    
                except Exception as e:
                    # Handle any unexpected errors
                    print(f"Error in perform_check: {str(e)}")
                    
                    try:
                        if progress_dialog and progress_dialog.isVisible():
                            progress_dialog.accept()
                    except:
                        pass
                    
                    # Create a minimal capabilities object with error info
                    error_capabilities = {
                        "name": model_name,
                        "error": str(e),
                        "check_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "source": "error"
                    }
                    
                    # Show dialog with the error information or fallback to message box
                    try:
                        show_capabilities_dialog(model_name, error_capabilities)
                    except Exception as dialog_error:
                        QMessageBox.critical(
                            dialog,
                            "Error Checking Capabilities",
                            f"An error occurred while checking capabilities: {str(e)}\n\nAdditional error: {str(dialog_error)}"
                        )
            
            # Use a timer to allow UI updates
            QTimer.singleShot(100, perform_check)
        
        # Function to show capabilities dialog
        def show_capabilities_dialog(model_name, capabilities):
            cap_dialog = QDialog(dialog)
            cap_dialog.setWindowTitle(f"Capabilities: {model_name}")
            cap_dialog.setMinimumWidth(600)
            cap_dialog.setMinimumHeight(500)
            
            cap_layout = QVBoxLayout(cap_dialog)
            
            # Create a scroll area for capabilities
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet("background-color: #1E2428;")
            
            # Container widget for scroll area
            scroll_content = QWidget()
            scroll_layout = QVBoxLayout(scroll_content)
            
            # Model name and basic info
            model_title = QLabel(f"<h2>{model_name}</h2>")
            model_title.setStyleSheet("color: #E9EDF0; font-weight: bold;")
            scroll_layout.addWidget(model_title)
            
            # Check time info
            check_time = capabilities.get("check_time", time.strftime("%Y-%m-%d %H:%M:%S"))
            time_label = QLabel(f"<b>Checked at:</b> {check_time}")
            time_label.setStyleSheet("color: #E9EDF0; font-size: 12px;")
            scroll_layout.addWidget(time_label)
            
            # Source info
            source = capabilities.get("source", "Unknown")
            source_label = QLabel(f"<b>Source:</b> {source}")
            source_label.setStyleSheet("color: #E9EDF0; font-size: 14px;")
            scroll_layout.addWidget(source_label)
            
            # Display any errors that occurred during capability checking
            if "error" in capabilities:
                error_frame = QFrame()
                error_frame.setFrameShape(QFrame.StyledPanel)
                error_frame.setStyleSheet("background-color: #5A1E1E; border-radius: 8px; padding: 10px;")
                error_layout = QVBoxLayout(error_frame)
                
                error_title = QLabel("<b>Error During Capability Check:</b>")
                error_title.setStyleSheet("color: #FF6B6B; font-size: 14px;")
                error_layout.addWidget(error_title)
                
                error_msg = QLabel(capabilities["error"])
                error_msg.setWordWrap(True)
                error_msg.setStyleSheet("color: #FFB6B6; font-size: 13px;")
                error_layout.addWidget(error_msg)
                
                error_note = QLabel("Note: Some capability information may still be available.")
                error_note.setStyleSheet("color: #FFB6B6; font-size: 12px; font-style: italic;")
                error_layout.addWidget(error_note)
                
                scroll_layout.addWidget(error_frame)
                
            # Display API errors if they occurred
            if "api_error" in capabilities:
                api_error_label = QLabel(f"<b>API Error:</b> {capabilities['api_error']}")
                api_error_label.setWordWrap(True)
                api_error_label.setStyleSheet("color: #FF9800; font-size: 13px;")
                scroll_layout.addWidget(api_error_label)
                
            # Display query errors if they occurred
            if "query_error" in capabilities:
                query_error_label = QLabel(f"<b>Query Error:</b> {capabilities['query_error']}")
                query_error_label.setWordWrap(True)
                query_error_label.setStyleSheet("color: #FF9800; font-size: 13px;")
                scroll_layout.addWidget(query_error_label)
            
            # Add separator
            separator = QFrame()
            separator.setFrameShape(QFrame.HLine)
            separator.setFrameShadow(QFrame.Sunken)
            separator.setStyleSheet("background-color: #2D383E;")
            scroll_layout.addWidget(separator)
            
            # Basic capabilities section
            try:
                basic_cap_label = QLabel("<h3>Basic Capabilities</h3>")
                basic_cap_label.setStyleSheet("color: #E9EDF0; margin-top: 10px;")
                scroll_layout.addWidget(basic_cap_label)
                
                # Create a grid for basic capabilities
                basic_grid = QGridLayout()
                basic_grid.setColumnStretch(1, 1)
                
                # Row counter for grid
                row = 0
                
                # Image generation
                try:
                    img_gen = capabilities.get("image_generation", False)
                    img_gen_label = QLabel("Image Generation:")
                    img_gen_label.setStyleSheet("color: #E9EDF0; font-size: 14px;")
                    img_gen_value = QLabel("✓ Supported" if img_gen else "✗ Not Supported")
                    img_gen_value.setStyleSheet(f"color: {'#4CAF50' if img_gen else '#F44336'}; font-size: 14px;")
                    basic_grid.addWidget(img_gen_label, row, 0)
                    basic_grid.addWidget(img_gen_value, row, 1)
                    row += 1
                except Exception as e:
                    print(f"Error displaying image generation capability: {str(e)}")
                
                # Vision capability
                try:
                    vision = capabilities.get("vision_capable", False)
                    vision_label = QLabel("Vision/Image Understanding:")
                    vision_label.setStyleSheet("color: #E9EDF0; font-size: 14px;")
                    vision_value = QLabel("✓ Supported" if vision else "✗ Not Supported")
                    vision_value.setStyleSheet(f"color: {'#4CAF50' if vision else '#F44336'}; font-size: 14px;")
                    basic_grid.addWidget(vision_label, row, 0)
                    basic_grid.addWidget(vision_value, row, 1)
                    row += 1
                except Exception as e:
                    print(f"Error displaying vision capability: {str(e)}")
                
                # Model family if available
                try:
                    if "model_family" in capabilities:
                        family_label = QLabel("Model Family:")
                        family_label.setStyleSheet("color: #E9EDF0; font-size: 14px;")
                        family_value = QLabel(str(capabilities["model_family"]))
                        family_value.setStyleSheet("color: #E9EDF0; font-size: 14px;")
                        basic_grid.addWidget(family_label, row, 0)
                        basic_grid.addWidget(family_value, row, 1)
                        row += 1
                except Exception as e:
                    print(f"Error displaying model family: {str(e)}")
                
                # Model type if available
                try:
                    if "model_type" in capabilities:
                        model_type_label = QLabel("Model Type:")
                        model_type_label.setStyleSheet("color: #E9EDF0; font-size: 14px;")
                        model_type_value = QLabel(str(capabilities["model_type"]))
                        model_type_value.setStyleSheet("color: #E9EDF0; font-size: 14px;")
                        basic_grid.addWidget(model_type_label, row, 0)
                        basic_grid.addWidget(model_type_value, row, 1)
                        row += 1
                except Exception as e:
                    print(f"Error displaying model type: {str(e)}")
                
                # Size information
                try:
                    size_label = QLabel("Model Size:")
                    size_label.setStyleSheet("color: #E9EDF0; font-size: 14px;")
                    
                    size_mb = capabilities.get("size_mb", 0)
                    if size_mb and size_mb >= 1000 and "size_gb" in capabilities:
                        size_text = f"{capabilities['size_gb']:.2f} GB"
                    elif size_mb and size_mb > 0:
                        size_text = f"{size_mb:.0f} MB"
                    else:
                        size_text = "Unknown"
                        
                    size_value = QLabel(size_text)
                    size_value.setStyleSheet("color: #E9EDF0; font-size: 14px;")
                    basic_grid.addWidget(size_label, row, 0)
                    basic_grid.addWidget(size_value, row, 1)
                    row += 1
                except Exception as e:
                    print(f"Error displaying size information: {str(e)}")
                
                # Base model if available
                try:
                    if "base_model" in capabilities and capabilities["base_model"]:
                        base_model_label = QLabel("Base Model:")
                        base_model_label.setStyleSheet("color: #E9EDF0; font-size: 14px;")
                        base_model_value = QLabel(str(capabilities["base_model"]))
                        base_model_value.setStyleSheet("color: #E9EDF0; font-size: 14px;")
                        basic_grid.addWidget(base_model_label, row, 0)
                        basic_grid.addWidget(base_model_value, row, 1)
                        row += 1
                except Exception as e:
                    print(f"Error displaying base model: {str(e)}")
                    
                # License if available
                try:
                    if "license" in capabilities and capabilities["license"]:
                        license_label = QLabel("License:")
                        license_label.setStyleSheet("color: #E9EDF0; font-size: 14px;")
                        license_value = QLabel(str(capabilities["license"]))
                        license_value.setStyleSheet("color: #E9EDF0; font-size: 14px;")
                        basic_grid.addWidget(license_label, row, 0)
                        basic_grid.addWidget(license_value, row, 1)
                        row += 1
                except Exception as e:
                    print(f"Error displaying license: {str(e)}")
            except Exception as e:
                print(f"Error setting up basic capabilities section: {str(e)}")
                # Add a fallback message if the grid setup fails
                fallback_label = QLabel("Could not display basic capabilities due to an error.")
                fallback_label.setStyleSheet("color: #FF9800; font-size: 14px;")
                scroll_layout.addWidget(fallback_label)
            
            # Add basic grid to layout
            scroll_layout.addLayout(basic_grid)
            
            # Add separator
            separator2 = QFrame()
            separator2.setFrameShape(QFrame.HLine)
            separator2.setFrameShadow(QFrame.Sunken)
            separator2.setStyleSheet("background-color: #2D383E;")
            scroll_layout.addWidget(separator2)
            
            # Advanced capabilities section
            try:
                if "parameters" in capabilities and capabilities["parameters"]:
                    # Ensure parameters is a dictionary
                    if isinstance(capabilities["parameters"], dict) and capabilities["parameters"]:
                        adv_cap_label = QLabel("<h3>Model Parameters</h3>")
                        adv_cap_label.setStyleSheet("color: #E9EDF0; margin-top: 10px;")
                        scroll_layout.addWidget(adv_cap_label)
                        
                        # Create a table for parameters
                        params_table = QTableWidget()
                        params_table.setColumnCount(2)
                        params_table.setHorizontalHeaderLabels(["Parameter", "Value"])
                        params_table.horizontalHeader().setStretchLastSection(True)
                        params_table.setStyleSheet("""
                            QTableWidget {
                                background-color: #1E2428;
                                color: #D1D7DB;
                                border: none;
                            }
                            QHeaderView::section {
                                background-color: #2A3942;
                                color: #E9EDF0;
                                padding: 5px;
                                border: none;
                            }
                            QTableWidget::item {
                                padding: 5px;
                            }
                        """)
                        
                        # Add parameters to table
                        try:
                            params = capabilities["parameters"]
                            params_table.setRowCount(len(params))
                            row = 0
                            for param, value in params.items():
                                try:
                                    # Convert param and value to strings to avoid any type issues
                                    param_str = str(param) if param is not None else "unknown"
                                    value_str = str(value) if value is not None else "unknown"
                                    
                                    # Create table items
                                    param_item = QTableWidgetItem(param_str)
                                    value_item = QTableWidgetItem(value_str)
                                    
                                    # Add to table
                                    params_table.setItem(row, 0, param_item)
                                    params_table.setItem(row, 1, value_item)
                                    row += 1
                                except Exception as item_error:
                                    print(f"Error adding parameter item: {str(item_error)}")
                                    continue
                            
                            # Resize table to fit content
                            params_table.resizeColumnsToContents()
                            params_table.setMinimumHeight(min(200, row * 30 + 40))  # Limit height
                            
                            scroll_layout.addWidget(params_table)
                        except Exception as params_error:
                            print(f"Error populating parameters table: {str(params_error)}")
                            params_error_label = QLabel("Could not display parameters due to an error.")
                            params_error_label.setStyleSheet("color: #FF9800; font-size: 13px;")
                            scroll_layout.addWidget(params_error_label)
            except Exception as e:
                print(f"Error setting up parameters section: {str(e)}")
            
            # Self-reported capabilities
            try:
                if "self_reported" in capabilities and capabilities["self_reported"]:
                    # Ensure self_reported is a string
                    if capabilities["self_reported"]:
                        self_report_label = QLabel("<h3>Self-Reported Capabilities</h3>")
                        self_report_label.setStyleSheet("color: #E9EDF0; margin-top: 10px;")
                        scroll_layout.addWidget(self_report_label)
                        
                        self_report_text = QTextEdit()
                        self_report_text.setReadOnly(True)
                        
                        # Convert to string if needed
                        self_report_content = str(capabilities["self_reported"])
                        self_report_text.setPlainText(self_report_content)
                        
                        self_report_text.setStyleSheet("""
                            QTextEdit {
                                background-color: #1E2428;
                                color: #D1D7DB;
                                border: none;
                                padding: 5px;
                                font-size: 14px;
                            }
                        """)
                        self_report_text.setMaximumHeight(100)
                        
                        scroll_layout.addWidget(self_report_text)
            except Exception as e:
                print(f"Error displaying self-reported capabilities: {str(e)}")
                try:
                    # Add a fallback message
                    self_report_error = QLabel("Could not display self-reported capabilities due to an error.")
                    self_report_error.setStyleSheet("color: #FF9800; font-size: 13px;")
                    scroll_layout.addWidget(self_report_error)
                except:
                    pass
            
            # Set scroll content and add to layout
            scroll.setWidget(scroll_content)
            cap_layout.addWidget(scroll)
            
            # Close button
            close_btn = QPushButton("Close")
            close_btn.setStyleSheet("""
                QPushButton {
                    background-color: #607D8B;
                    color: white;
                    border-radius: 4px;
                    padding: 8px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #455A64;
                }
            """)
            close_btn.clicked.connect(cap_dialog.accept)
            
            cap_layout.addWidget(close_btn)
            
            # Show dialog
            cap_dialog.exec_()
        
        switch_model_btn.clicked.connect(switch_model)
        check_capabilities_btn.clicked.connect(check_model_capabilities)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(switch_model_btn)
        button_layout.addWidget(check_capabilities_btn)
        
        # Add widgets to model tab
        model_layout.addWidget(current_model_frame)
        model_layout.addWidget(models_label)
        model_layout.addWidget(models_list)
        model_layout.addLayout(button_layout)
        
        # Scan for local models tab
        scan_layout = QVBoxLayout(scan_tab)
        
        # Scan instructions
        scan_info = QLabel("Scan your system for local AI models. This will search common directories where models are stored.")
        scan_info.setWordWrap(True)
        scan_info.setStyleSheet("color: #E9EDF0; font-size: 14px;")
        
        # Custom path input
        custom_path_layout = QHBoxLayout()
        custom_path_label = QLabel("Custom Path:")
        custom_path_label.setStyleSheet("color: #E9EDF0; font-size: 14px;")
        custom_path_input = QLineEdit()
        custom_path_input.setPlaceholderText("Optional: Enter a custom directory to scan")
        custom_path_input.setStyleSheet("""
            QLineEdit {
                background-color: #1E2428;
                color: #D1D7DB;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }
        """)
        browse_btn = QPushButton("Browse")
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #455A64;
            }
        """)
        
        # Connect browse button
        def browse_directory():
            dir_path = QFileDialog.getExistingDirectory(dialog, "Select Directory to Scan")
            if dir_path:
                custom_path_input.setText(dir_path)
        
        browse_btn.clicked.connect(browse_directory)
        
        custom_path_layout.addWidget(custom_path_label)
        custom_path_layout.addWidget(custom_path_input)
        custom_path_layout.addWidget(browse_btn)
        
        # Scan results
        scan_results_label = QLabel("Scan Results:")
        scan_results_label.setStyleSheet("color: #E9EDF0; font-size: 14px; margin-top: 10px;")
        
        scan_results = QTextEdit()
        scan_results.setReadOnly(True)
        scan_results.setStyleSheet("""
            QTextEdit {
                background-color: #1E2428;
                color: #D1D7DB;
                border-radius: 4px;
                padding: 5px;
                font-size: 14px;
                font-family: monospace;
            }
        """)
        
        # Progress bar
        scan_progress = QProgressBar()
        scan_progress.setRange(0, 100)
        scan_progress.setValue(0)
        scan_progress.setVisible(False)
        scan_progress.setStyleSheet("""
            QProgressBar {
                background-color: #1E2428;
                color: white;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #00A884;
                border-radius: 4px;
            }
        """)
        
        # Scan button
        scan_btn = QPushButton("Scan for Local Models")
        scan_btn.setStyleSheet("""
            QPushButton {
                background-color: #00A884;
                color: white;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #009673;
            }
        """)
        
        # Connect scan button
        def scan_for_models():
            # Update UI
            scan_btn.setEnabled(False)
            scan_results.clear()
            scan_progress.setVisible(True)
            scan_progress.setValue(10)
            scan_results.append("Scanning for local AI models...\n")
            
            # Get custom path if provided
            custom_path = custom_path_input.text().strip()
            if custom_path and os.path.exists(custom_path):
                self.chatbot.local_model_paths.append(custom_path)
                scan_results.append(f"Added custom path: {custom_path}\n")
            
            # Use a timer to allow UI updates
            def perform_scan():
                # Clear existing models
                old_models = set(self.chatbot.models)
                scan_progress.setValue(30)
                
                # Scan for models
                self.chatbot.scan_local_model_directories()
                scan_progress.setValue(70)
                
                # Update models list
                models_list.clear()
                for model in self.chatbot.models:
                    item = QListWidgetItem(model)
                    if model in self.chatbot.model_capabilities and self.chatbot.model_capabilities[model].get("image_generation", False):
                        item.setIcon(QIcon.fromTheme("image"))
                        item.setToolTip("Supports image generation")
                    models_list.setCurrentItem(item) if model == self.chatbot.model else None
                    models_list.addItem(item)
                
                # Show results
                scan_progress.setValue(90)
                new_models = set(self.chatbot.models) - old_models
                scan_results.append(f"Scan complete! Found {len(self.chatbot.models)} models in total.\n")
                
                # Group models by source
                ollama_models = []
                local_file_models = []
                other_models = []
                
                for model in self.chatbot.models:
                    if model in self.chatbot.model_capabilities:
                        source = self.chatbot.model_capabilities[model].get("source", "")
                        if source in ["ollama_api", "ollama_cli"]:
                            ollama_models.append(model)
                        elif source == "file_scan":
                            local_file_models.append(model)
                        else:
                            other_models.append(model)
                
                # Display newly discovered models with more details
                if new_models:
                    scan_results.append("Newly discovered models:")
                    for model in sorted(new_models):
                        model_info = ""
                        if model in self.chatbot.model_capabilities:
                            capabilities = self.chatbot.model_capabilities[model]
                            
                            # Add source info
                            source = capabilities.get("source", "")
                            if source == "ollama_api":
                                model_info += " [Ollama API]"
                            elif source == "ollama_cli":
                                model_info += " [Ollama CLI]"
                            elif source == "file_scan":
                                model_info += " [Local File]"
                                
                                # Add path for file-based models
                                if "local_path" in capabilities:
                                    model_info += f" - {capabilities['local_path']}"
                            
                            # Add image generation capability
                            if capabilities.get("image_generation", False):
                                model_info += " ✓ Images"
                            
                            # Add size info if available
                            if "size_mb" in capabilities and capabilities["size_mb"] > 0:
                                size_mb = capabilities["size_mb"]
                                if size_mb >= 1000:
                                    size_gb = size_mb / 1024
                                    model_info += f" - {size_gb:.1f} GB"
                                else:
                                    model_info += f" - {size_mb:.0f} MB"
                        
                        scan_results.append(f"- {model}{model_info}")
                else:
                    scan_results.append("No new models discovered.")
                
                # Display summary by source
                scan_results.append("\nModel Summary:")
                if ollama_models:
                    scan_results.append(f"- Ollama Models: {len(ollama_models)}")
                if local_file_models:
                    scan_results.append(f"- Local File Models: {len(local_file_models)}")
                if other_models:
                    scan_results.append(f"- Other Models: {len(other_models)}")
                
                # Reset UI
                scan_progress.setValue(100)
                scan_btn.setEnabled(True)
                
                # Use a timer to hide progress bar after a delay
                QTimer.singleShot(2000, lambda: scan_progress.setVisible(False))
            
            # Use a timer to allow UI updates
            QTimer.singleShot(100, perform_scan)
        
        scan_btn.clicked.connect(scan_for_models)
        
        # Add widgets to scan tab
        scan_layout.addWidget(scan_info)
        scan_layout.addLayout(custom_path_layout)
        scan_layout.addWidget(scan_results_label)
        scan_layout.addWidget(scan_results)
        scan_layout.addWidget(scan_progress)
        scan_layout.addWidget(scan_btn)
        
        # Add tabs to tab widget
        tabs.addTab(model_tab, "Model Selection")
        tabs.addTab(scan_tab, "Scan for Models")
        
        # Add tab widget to main layout
        layout.addWidget(tabs)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #455A64;
            }
        """)
        close_btn.clicked.connect(dialog.accept)
        
        layout.addWidget(close_btn)
        
        # Show dialog
        dialog.exec_()
    
    def generate_ai_image(self, prompt, size, dialog):
        """Generate an image using the AI model"""
        # Set processing state
        self.set_processing_state(True)
        
        # Add user message with the prompt
        self.add_user_message(f"Generate an image: {prompt}")
        
        # Generate the image
        img_path, error = self.chatbot.generate_image(prompt, size)
        
        if img_path:
            # Add the generated image to the chat
            self.add_image_message(img_path, f"AI generated image: {prompt}", is_user=False)
            # Close the dialog
            dialog.accept()
        else:
            # Show error message
            error_msg = error if error else "Failed to generate image. Please try again."
            self.add_system_message(error_msg, is_error=True)
            # Update dialog to show error
            dialog.findChild(QLabel, "progress_label").setText("Error: " + error_msg)
            dialog.findChild(QLabel, "progress_label").setStyleSheet("color: red;")
            dialog.findChild(QProgressBar).setVisible(False)
            dialog.findChild(QPushButton, "generate_btn").setEnabled(True)
            dialog.findChild(QTextEdit).setEnabled(True)
            dialog.findChild(QComboBox).setEnabled(True)
        
        # Reset processing state
        self.set_processing_state(False)
        
    def analyze_image(self, image_path):
        """Analyze an image using AI"""
        try:
            # Check if OpenCV is available
            import cv2
            import numpy as np
            
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                self.add_system_message(f"Error: Could not read image {image_path}", is_error=True)
                return
            
            # Get basic image info
            height, width, channels = img.shape
            image_info = f"Image dimensions: {width}x{height}, {channels} channels"            
            
            # Prepare prompt for AI
            prompt = f"I've uploaded an image. Please describe what you see in this image. {image_info}"
            
            # Set processing state
            self.set_processing_state(True)
            
            # Get AI response
            system_prompt = "You are a helpful assistant that can analyze images. Describe what you see in the image based on the information provided."
            response = self.chatbot.generate_response(prompt, system_prompt)
            
            # Add AI response
            self.add_ai_message(response)
            
        except ImportError:
            self.add_system_message("OpenCV is not installed. Cannot analyze image.", is_error=True)
            self.add_ai_message("I see you've shared an image, but I don't have the necessary libraries installed to analyze it in detail. If you'd like me to comment on this image, please describe what it shows.")
        except Exception as e:
            self.add_system_message(f"Error analyzing image: {str(e)}", is_error=True)
        finally:
            # Reset processing state
            self.set_processing_state(False)
    
    def add_image_message(self, image_path, caption=""):
        """Add an image message to the chat"""
        # Create conversation if none exists
        if not self.current_conversation_id:
            self.new_conversation()
        
        # Add to database
        if self.current_conversation_id:
            self.db.add_message(self.current_conversation_id, "user", caption, image_path)
        
        # Legacy: Add to history manager
        self.chat_history_manager.add_message("user", caption, image_path)
        
        # Add to UI
        image_bubble = ImageMessageBubble(image_path, caption, is_user=True)
        self.chat_layout.addWidget(image_bubble)
        
        # Scroll to bottom
        QApplication.processEvents()  # Force update to get correct scroll height
        scroll_area = self.chat_container.parent()
        if hasattr(scroll_area, 'verticalScrollBar'):
            scroll_area.verticalScrollBar().setValue(
                scroll_area.verticalScrollBar().maximum()
            )
    
    def add_user_message(self, content):
        """Add a user message to the chat"""
        # Create conversation if none exists
        if not self.current_conversation_id:
            self.new_conversation()
        
        # Add to database
        if self.current_conversation_id:
            self.db.add_message(self.current_conversation_id, "user", content)
        
        # Legacy: Add to history manager
        self.chat_history_manager.add_message("user", content)
        
        # Add to UI
        message_bubble = MessageBubble(content, is_user=True)
        self.chat_layout.addWidget(message_bubble)
        
        # Scroll to bottom
        QApplication.processEvents()  # Force update to get correct scroll height
        scroll_area = self.chat_container.parent()
        if hasattr(scroll_area, 'verticalScrollBar'):
            scroll_area.verticalScrollBar().setValue(
                scroll_area.verticalScrollBar().maximum()
            )
    
    def add_ai_message(self, content):
        """Add an AI message to the chat"""
        # Create conversation if none exists
        if not self.current_conversation_id:
            self.new_conversation()
        
        # Add to database
        if self.current_conversation_id:
            self.db.add_message(self.current_conversation_id, "ai", content)
        
        # Legacy: Add to history manager
        self.chat_history_manager.add_message("ai", content)
        
        # Add to UI
        message_bubble = MessageBubble(content, is_user=False)
        self.chat_layout.addWidget(message_bubble)
        
        # Scroll to bottom
        QApplication.processEvents()  # Force update to get correct scroll height
        scroll_area = self.chat_container.parent()
        if hasattr(scroll_area, 'verticalScrollBar'):
            scroll_area.verticalScrollBar().setValue(
                scroll_area.verticalScrollBar().maximum()
            )
    
    def add_system_message(self, content, is_error=False):
        """Add a system message to the chat"""
        # Create conversation if none exists
        if not self.current_conversation_id:
            self.new_conversation()
        
        # Add to database
        if self.current_conversation_id:
            self.db.add_message(self.current_conversation_id, "system", content)
        
        # Legacy: Add to history manager
        self.chat_history_manager.add_message("system", content)
        
        # Add to UI
        system_bubble = SystemMessageBubble(content, is_error)
        self.chat_layout.addWidget(system_bubble)
        
        # Scroll to bottom
        QApplication.processEvents()  # Force update to get correct scroll height
        scroll_area = self.chat_container.parent()
        if hasattr(scroll_area, 'verticalScrollBar'):
            scroll_area.verticalScrollBar().setValue(
                scroll_area.verticalScrollBar().maximum()
            )
    
    def load_conversations(self):
        """Load conversations from database"""
        self.conversations = self.db.get_conversations()
        
        # Update conversation selector
        if hasattr(self, 'conversation_selector'):
            self.conversation_selector.clear()
            
            for conversation in self.conversations:
                title = conversation["title"]
                self.conversation_selector.addItem(title, conversation["id"])
            
            # Add "New Conversation" option
            self.conversation_selector.addItem("+ New Conversation", -1)
            
            # Select first conversation or create new one if none exists
            if self.conversations:
                self.current_conversation_id = self.conversations[0]["id"]
                self.conversation_selector.setCurrentIndex(0)
                self.chatbot.current_mood = self.conversations[0]["ai_mood"]
                self.update_mood_indicator()
            else:
                self.new_conversation()
    
    def new_conversation(self):
        """Create a new conversation"""
        title = f"Conversation {len(self.conversations) + 1}"
        conversation_id = self.db.create_conversation(title, self.chatbot.current_mood)
        
        if conversation_id:
            self.current_conversation_id = conversation_id
            
            # Add to conversations list
            self.conversations.insert(0, {
                "id": conversation_id,
                "title": title,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "ai_mood": self.chatbot.current_mood
            })
            
            # Update selector
            self.conversation_selector.blockSignals(True)
            self.conversation_selector.insertItem(0, title, conversation_id)
            self.conversation_selector.setCurrentIndex(0)
            self.conversation_selector.blockSignals(False)
            
            # Clear chat display
            self.clear_chat_display()
            
            # Add welcome message
            self.add_system_message(f"New conversation started with {AI_MOODS[self.chatbot.current_mood]['name']} mood.")
    
    def change_conversation(self, index):
        """Change to a different conversation"""
        conversation_id = self.conversation_selector.itemData(index)
        
        # If "New Conversation" selected
        if conversation_id == -1:
            self.new_conversation()
            return
        
        # Set current conversation
        self.current_conversation_id = conversation_id
        
        # Find conversation in list
        for conversation in self.conversations:
            if conversation["id"] == conversation_id:
                # Update AI mood
                self.chatbot.current_mood = conversation["ai_mood"]
                
                # Update mood selector
                mood_index = self.mood_selector.findData(conversation["ai_mood"])
                if mood_index >= 0:
                    self.mood_selector.blockSignals(True)
                    self.mood_selector.setCurrentIndex(mood_index)
                    self.mood_selector.blockSignals(False)
                    self.update_mood_indicator()
                break
        
        # Display messages
        self.display_conversation_messages()
    
    def delete_current_conversation(self):
        """Delete the current conversation"""
        if not self.current_conversation_id:
            return
            
        # Confirm deletion
        reply = QMessageBox.question(
            self, 
            "Delete Conversation", 
            "Are you sure you want to delete this conversation? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Delete from database
            success = self.db.delete_conversation(self.current_conversation_id)
            
            if success:
                # Remove from selector
                current_index = self.conversation_selector.currentIndex()
                self.conversation_selector.removeItem(current_index)
                
                # Remove from list
                self.conversations = [c for c in self.conversations if c["id"] != self.current_conversation_id]
                
                # Select another conversation or create new one
                if self.conversations:
                    # Select the next available conversation
                    if current_index >= len(self.conversations):
                        current_index = 0
                    self.conversation_selector.setCurrentIndex(current_index)
                else:
                    self.new_conversation()
    
    def clear_chat_display(self):
        """Clear the chat display without affecting the database"""
        # Clear existing widgets
        while self.chat_layout.count():
            item = self.chat_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def display_conversation_messages(self):
        """Display messages for the current conversation"""
        if not self.current_conversation_id:
            return
            
        # Clear display
        self.clear_chat_display()
        
        # Get messages from database
        messages = self.db.get_messages(self.current_conversation_id)
        
        # Add messages to display
        for message in messages:
            role = message.get("role", "system")
            content = message.get("content", "")
            image_path = message.get("image_path")
            
            if role == "user":
                if image_path and os.path.exists(image_path):
                    # Image message
                    image_bubble = ImageMessageBubble(image_path, content, is_user=True)
                    self.chat_layout.addWidget(image_bubble)
                else:
                    # Regular text message
                    message_bubble = MessageBubble(content, is_user=True)
                    self.chat_layout.addWidget(message_bubble)
            elif role == "ai":
                message_bubble = MessageBubble(content, is_user=False)
                self.chat_layout.addWidget(message_bubble)
            elif role == "system":
                # Determine if it's an error message
                is_error = "error" in content.lower() or "failed" in content.lower()
                system_bubble = SystemMessageBubble(content, is_error)
                self.chat_layout.addWidget(system_bubble)
        
        # Scroll to bottom
        QApplication.processEvents()  # Force update to get correct scroll height
        scroll_area = self.chat_container.parent()
        if hasattr(scroll_area, 'verticalScrollBar'):
            scroll_area.verticalScrollBar().setValue(
                scroll_area.verticalScrollBar().maximum()
            )
    
    def display_chat_history(self):
        """Display the chat history (legacy method)"""
        # If we have a current conversation, use that instead
        if self.current_conversation_id:
            self.display_conversation_messages()
            return
            
        # Legacy code for JSON-based history
        # Clear existing widgets
        while self.chat_layout.count():
            item = self.chat_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add messages from history
        for message in self.chat_history_manager.get_history():
            role = message.get("role", "system")
            content = message.get("content", "")
            image_path = message.get("image_path")
            
            if role == "user":
                if image_path and os.path.exists(image_path):
                    # Image message
                    image_bubble = ImageMessageBubble(image_path, content, is_user=True)
                    self.chat_layout.addWidget(image_bubble)
                else:
                    # Regular text message
                    message_bubble = MessageBubble(content, is_user=True)
                    self.chat_layout.addWidget(message_bubble)
            elif role == "ai":
                message_bubble = MessageBubble(content, is_user=False)
                self.chat_layout.addWidget(message_bubble)
            elif role == "system":
                # Determine if it's an error message
                is_error = "error" in content.lower() or "failed" in content.lower()
                system_bubble = SystemMessageBubble(content, is_error)
                self.chat_layout.addWidget(system_bubble)
        
        # Scroll to bottom
        QApplication.processEvents()  # Force update to get correct scroll height
        scroll_area = self.chat_container.parent()
        if hasattr(scroll_area, 'verticalScrollBar'):
            scroll_area.verticalScrollBar().setValue(
                scroll_area.verticalScrollBar().maximum()
            )
    
    def clear_chat_history(self):
        """Clear the chat history"""
        if self.current_conversation_id:
            # Delete all messages for this conversation
            cursor = self.db.conn.cursor()
            cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (self.current_conversation_id,))
            self.db.conn.commit()
            
            # Clear display
            self.clear_chat_display()
            
            # Add system message
            self.add_system_message("Chat history cleared.")
        else:
            # Legacy: Clear JSON-based history
            self.chat_history_manager.clear_history()
            
            # Clear existing widgets
            while self.chat_layout.count():
                item = self.chat_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            # Add system message
            self.add_system_message("Chat history cleared.")
    
    def send_message(self):
        """Send a message to the AI"""
        message = self.chat_input.text().strip()
        if not message:
            return
            
        # Add user message to chat
        self.add_user_message(message)
        self.chat_input.clear()
        
        # Change send button to a square stop button
        self.set_processing_state(True)
        QApplication.processEvents()  # Force UI update
        
        try:
            # Check if it's a command to restart the AI
            if message.lower() in ["restart ai", "restart", "reconnect"]:
                system_msg = "Attempting to reconnect to AI service..."
                self.add_system_message(system_msg)
                
                if self.chatbot.check_ollama_status():
                    system_msg = f"Successfully reconnected! Using model: {self.chatbot.model}"
                    self.add_system_message(system_msg)
                    
                    # Update UI
                    self.model_status_label.setText("✓ Online")
                    self.model_status_label.setStyleSheet("color: #4ec9b0;")
                    self.refresh_models()
                else:
                    system_msg = "Failed to reconnect. Remaining in offline mode."
                    self.add_system_message(system_msg, is_error=True)
            else:
                # Get response from AI using the current mood's system prompt
                system_prompt = AI_MOODS[self.chatbot.current_mood]["system_prompt"]
                response = self.chatbot.generate_response(message, system_prompt)
                
                # Add AI response to chat
                self.add_ai_message(response)
        except Exception as e:
            # Handle any errors
            self.add_system_message(f"Error: {str(e)}", is_error=True)
        finally:
            # Change button back to send arrow
            self.set_processing_state(False)
        
    def load_todos(self):
        """Load todos from file"""
        if os.path.exists(TODO_FILE):
            try:
                with open(TODO_FILE, "r") as f:
                    self.todos = json.load(f)
                    
                # Add widgets for each todo
                for todo in self.todos:
                    self.add_todo_widget(todo)
            except Exception as e:
                print(f"Error loading todos: {e}")
                
    def save_todos(self):
        """Save todos to file"""
        try:
            with open(TODO_FILE, "w") as f:
                json.dump(self.todos, f, indent=2)
        except Exception as e:
            print(f"Error saving todos: {e}")
            
    def closeEvent(self, event):
        """Handle window close event"""
        self.save_todos()
        
        # Close database connection
        if hasattr(self, 'db'):
            self.db.close()
            
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = ChatMateApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
