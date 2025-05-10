#!/usr/bin/env python3
import sys
import json
import os
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QLineEdit, 
                            QListWidget, QListWidgetItem, QTabWidget, QTextEdit,
                            QScrollArea, QFrame, QSplitter, QMessageBox, QComboBox,
                            QDialog, QCheckBox)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QIcon

# Constants
TODO_FILE = "todos.json"
CATEGORIES = ["To Do", "Ongoing", "Done", "Waiting", "Someday"]

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
        # Select the 5 smallest models based on the ollama list output
        self.models = ["llama3.2", "phi3", "mistral", "codellama", "gemma2"]
        self.available_models = []
        self.current_model_index = 0
        self.model = self.models[self.current_model_index] if self.models else None
        self.offline_mode = False
        self.model_sizes = {}
        self.online_api = OnlineAIAPI()
        self.use_online_api = False
        self.get_available_models()
        
    def get_available_models(self):
        """Get list of available models from Ollama"""
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
                    
                    for model in models_data:
                        name = model.get('name', '').split(':')[0]
                        size = model.get('size', 0)
                        self.model_sizes[name] = size
                        self.available_models.append(name)
                    
                    # Sort models by size (smallest first)
                    self.available_models.sort(key=lambda x: self.model_sizes.get(x, float('inf')))
                    
                    # Take the 5 smallest models or fewer if less are available
                    self.models = self.available_models[:5] if self.available_models else self.models
                    
                    if self.models:
                        self.model = self.models[0]
                        self.current_model_index = 0
                        self.offline_mode = False
                        return True
                except json.JSONDecodeError:
                    # Fallback to text parsing if JSON fails
                    return self.parse_ollama_list_text(result.stdout)
            else:
                return self.check_ollama_status()
                
        except (subprocess.SubprocessError, FileNotFoundError):
            return self.check_ollama_status()
    
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
        # Simple rule-based responses for offline mode
        prompt_lower = prompt.lower()
        
        if "hello" in prompt_lower or "hi" in prompt_lower:
            return "Hello! I'm currently in offline mode, but I'm happy to help with basic todo management."
        
        if "help" in prompt_lower:
            return "I can help you manage your tasks. You can add new tasks, mark them as complete, or organize them into different categories."
        
        if "task" in prompt_lower or "todo" in prompt_lower:
            return "To manage your tasks, use the Todo List tab. You can add new tasks, change their status, or delete them as needed."
        
        if "thank" in prompt_lower:
            return "You're welcome! Let me know if you need anything else."
        
        # Default response
        return "I'm currently in offline mode. The Ollama AI service isn't available right now. You can still use all the todo list features, but AI assistance is limited."

class TodoItem(QWidget):
    """Custom widget for todo items"""
    deleted = pyqtSignal(object)
    status_changed = pyqtSignal(object, str)
    
    def __init__(self, todo, parent=None):
        super().__init__(parent)
        self.todo = todo
        
        # Main layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
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
    
    def __init__(self, parent=None, chatbot=None):
        super().__init__(parent)
        self.chatbot = chatbot
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
        if self.chatbot and self.chatbot.use_online_api:
            self.use_online_checkbox.setChecked(True)
        
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
        if not self.chatbot:
            self.reject()
            return
            
        api_key = self.key_input.text().strip()
        provider = self.provider_combo.currentText()
        model = self.model_combo.currentText()
        use_online = self.use_online_checkbox.isChecked()
        
        if not api_key:
            QMessageBox.warning(self, "Warning", "API key cannot be empty")
            return
            
        # Set the API key and provider
        success = self.chatbot.online_api.set_api_key(api_key, provider)
        
        if success:
            # Set the model
            self.chatbot.online_api.set_model(model)
            
            # Set whether to use online API
            self.chatbot.use_online_api = use_online
            
            self.accept()
        else:
            QMessageBox.critical(self, "Error", f"Failed to configure {provider} API")

class MessageBubble(QWidget):
    """Custom widget for chat message bubbles"""
    
    def __init__(self, message, is_user=True, parent=None):
        super().__init__(parent)
        self.message = message
        self.is_user = is_user
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        # Main layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Create bubble widget
        bubble = QFrame(self)
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(12, 8, 12, 8)
        
        # Message text
        message_label = QLabel(self.message)
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        # Add timestamp if needed
        # timestamp_label = QLabel(datetime.now().strftime("%H:%M"))
        # timestamp_label.setStyleSheet("color: rgba(255, 255, 255, 0.6); font-size: 10px;")
        # timestamp_label.setAlignment(Qt.AlignRight)
        
        # Add widgets to bubble layout
        bubble_layout.addWidget(message_label)
        # bubble_layout.addWidget(timestamp_label)
        
        # Set alignment based on sender
        if self.is_user:
            layout.addStretch(1)
            layout.addWidget(bubble)
            bubble.setStyleSheet("""
                QFrame {
                    background-color: #128C7E; /* WhatsApp green */
                    color: white;
                    border-radius: 10px;
                    border-bottom-right-radius: 2px;
                }
                QLabel {
                    color: white;
                    font-size: 14px;
                }
            """)
        else:
            layout.addWidget(bubble)
            layout.addStretch(1)
            bubble.setStyleSheet("""
                QFrame {
                    background-color: #262D31; /* WhatsApp dark gray */
                    color: white;
                    border-radius: 10px;
                    border-bottom-left-radius: 2px;
                }
                QLabel {
                    color: white;
                    font-size: 14px;
                }
            """)

class SystemMessageBubble(QWidget):
    """Custom widget for system message bubbles"""
    
    def __init__(self, message, is_error=False, parent=None):
        super().__init__(parent)
        self.message = message
        self.is_error = is_error
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        # Main layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 2, 10, 2)
        
        # Create bubble widget
        bubble = QFrame(self)
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(12, 6, 12, 6)
        
        # Message text
        message_label = QLabel(self.message)
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        # Add widgets to bubble layout
        bubble_layout.addWidget(message_label)
        
        # Center the bubble
        layout.addStretch(1)
        layout.addWidget(bubble)
        layout.addStretch(1)
        
        # Style based on error status
        if self.is_error:
            bubble.setStyleSheet("""
                QFrame {
                    background-color: rgba(255, 107, 107, 0.7); /* Light red */
                    color: white;
                    border-radius: 10px;
                }
                QLabel {
                    color: white;
                    font-size: 12px;
                }
            """)
        else:
            bubble.setStyleSheet("""
                QFrame {
                    background-color: rgba(78, 201, 176, 0.7); /* Light teal */
                    color: white;
                    border-radius: 10px;
                }
                QLabel {
                    color: white;
                    font-size: 12px;
                }
            """)

class ImageMessageBubble(QWidget):
    """Custom widget for image message bubbles"""
    
    def __init__(self, image_path, caption="", is_user=True, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.caption = caption
        self.is_user = is_user
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        # Main layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Create bubble widget
        bubble = QFrame(self)
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(8, 8, 8, 8)
        
        # Image label
        from PyQt5.QtGui import QPixmap
        image_label = QLabel()
        pixmap = QPixmap(self.image_path)
        # Scale image to a reasonable size while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(300, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_label.setPixmap(scaled_pixmap)
        
        # Caption text if provided
        if self.caption:
            caption_label = QLabel(self.caption)
            caption_label.setWordWrap(True)
            caption_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            bubble_layout.addWidget(caption_label)
        
        # Add widgets to bubble layout
        bubble_layout.addWidget(image_label)
        
        # Set alignment based on sender
        if self.is_user:
            layout.addStretch(1)
            layout.addWidget(bubble)
            bubble.setStyleSheet("""
                QFrame {
                    background-color: #128C7E; /* WhatsApp green */
                    color: white;
                    border-radius: 10px;
                    border-bottom-right-radius: 2px;
                }
                QLabel {
                    color: white;
                    font-size: 14px;
                }
            """)
        else:
            layout.addWidget(bubble)
            layout.addStretch(1)
            bubble.setStyleSheet("""
                QFrame {
                    background-color: #262D31; /* WhatsApp dark gray */
                    color: white;
                    border-radius: 10px;
                    border-bottom-left-radius: 2px;
                }
                QLabel {
                    color: white;
                    font-size: 14px;
                }
            """)

class ChatHistory:
    """Class to store and manage chat history"""
    
    def __init__(self, max_history=10):
        self.max_history = max_history
        self.history = []
        self.history_file = "chat_history.json"
        self.load_history()
    
    def add_message(self, role, content, image_path=None):
        """Add a message to the history"""
        message = {
            "role": role,  # 'user', 'ai', or 'system'
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path  # Path to image if this is an image message
        }
        
        self.history.append(message)
        
        # Trim history if it exceeds max size
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            
        # Save history
        self.save_history()
        
        return message
    
    def get_history(self):
        """Get the full chat history"""
        return self.history
    
    def clear_history(self):
        """Clear the chat history"""
        self.history = []
        self.save_history()
    
    def save_history(self):
        """Save chat history to file"""
        try:
            with open(self.history_file, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Error saving chat history: {e}")
    
    def load_history(self):
        """Load chat history from file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    self.history = json.load(f)
                    
                    # Ensure we don't exceed max history
                    if len(self.history) > self.max_history:
                        self.history = self.history[-self.max_history:]
            except Exception as e:
                print(f"Error loading chat history: {e}")

class ChatMateApp(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.todos = []
        self.chatbot = ChatbotAPI()
        self.chat_history_manager = ChatHistory(max_history=10)
        self.init_ui()
        self.load_todos()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("ChatMate")
        self.setMinimumSize(800, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create todo tab
        todo_widget = QWidget()
        todo_layout = QVBoxLayout(todo_widget)
        
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
        
        # Create a tab for each category
        self.category_lists = {}
        for category in CATEGORIES:
            list_widget = QListWidget()
            list_widget.setSpacing(5)
            self.category_lists[category] = list_widget
            self.category_tabs.addTab(list_widget, f"{category} (0)")
        
        # Add widgets to todo layout
        todo_layout.addWidget(title_label)
        todo_layout.addLayout(input_layout)
        todo_layout.addWidget(self.category_tabs)
        
        # Create chat tab
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        
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
        
        clear_history_btn = QPushButton("Clear History")
        clear_history_btn.clicked.connect(self.clear_chat_history)
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_selector, 1)  # 1 = stretch factor
        model_layout.addWidget(model_status)
        model_layout.addWidget(refresh_btn)
        model_layout.addWidget(online_api_btn)
        model_layout.addWidget(self.online_api_label)
        model_layout.addWidget(clear_history_btn)
        
        # Chat history display - using a scroll area with widgets instead of QTextEdit
        chat_scroll = QScrollArea()
        chat_scroll.setWidgetResizable(True)
        chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        chat_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Container for chat messages
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setSpacing(10)
        self.chat_layout.setContentsMargins(0, 10, 0, 10)
        
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
        
        # Chat input area with WhatsApp-like styling
        chat_input_container = QWidget()
        chat_input_container.setStyleSheet("""
            QWidget {
                background-color: #1E2428; /* WhatsApp input area */
                border-top: 1px solid #2D383E;
            }
        """)
        chat_input_layout = QHBoxLayout(chat_input_container)
        chat_input_layout.setContentsMargins(10, 10, 10, 10)
        
        # Image upload button
        upload_btn = QPushButton()
        upload_btn.setIcon(QIcon.fromTheme("document-open"))
        upload_btn.setToolTip("Upload an image")
        upload_btn.setFixedSize(40, 40)
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
        
        # Chat input field
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type a message...")
        self.chat_input.returnPressed.connect(self.send_message)
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
        send_btn = QPushButton()
        send_btn.setIcon(QIcon.fromTheme("document-send"))
        send_btn.setToolTip("Send message")
        send_btn.setFixedSize(40, 40)
        send_btn.clicked.connect(self.send_message)
        send_btn.setStyleSheet("""
            QPushButton {
                background-color: #00A884; /* WhatsApp accent green */
                border-radius: 20px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #128C7E;
            }
        """)
        
        chat_input_layout.addWidget(upload_btn)
        chat_input_layout.addWidget(self.chat_input)
        chat_input_layout.addWidget(send_btn)
        
        # Add all widgets to chat layout
        chat_title = QLabel("ChatMate Assistant")
        chat_title.setStyleSheet("""
            font-size: 18px; 
            font-weight: bold; 
            color: #d7ba7d; 
            border-bottom: 1px solid #3c3c3c;
            padding-bottom: 5px;
            margin-bottom: 10px;
        """)
        
        chat_layout.addWidget(chat_title)
        chat_layout.addLayout(model_layout)
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
                self.chat_history.append(f"<span style='color:#4ec9b0;'><b>System:</b> Switched to model: {model_name}</span>")
                self.chat_history.append("")
            else:
                self.chat_history.append(f"<span style='color:#ff6b6b;'><b>System:</b> Failed to switch to model: {model_name}</span>")
                self.chat_history.append("")
                
            # Scroll to bottom
            self.chat_history.verticalScrollBar().setValue(
                self.chat_history.verticalScrollBar().maximum()
            )
    
    def configure_online_api(self):
        """Open the dialog to configure online AI API"""
        dialog = OnlineAPIDialog(self, self.chatbot)
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            # Update the online API indicator
            if self.chatbot.use_online_api and self.chatbot.online_api.available:
                provider = self.chatbot.online_api.provider.capitalize()
                model = self.chatbot.online_api.current_model
                self.online_api_label.setText(f"☁️ {provider}")
                self.online_api_label.setStyleSheet("color: #4ec9b0;")
                
                # Add message to chat history
                self.chat_history.append(f"<span style='color:#4ec9b0;'><b>System:</b> Connected to {provider} API using model {model}.</span>")
                self.chat_history.append("")
                
                # Save to chat history manager
                self.chat_history_manager.add_message("system", f"Connected to {provider} API using model {model}.")
            else:
                self.online_api_label.setText("☁️ Off")
                self.online_api_label.setStyleSheet("color: #6c6c6c;")
    
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
        from PyQt5.QtWidgets import QFileDialog
        
        # Open file dialog to select an image
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        file_dialog.setViewMode(QFileDialog.Detail)
        
        if file_dialog.exec_():
            image_path = file_dialog.selectedFiles()[0]
            if image_path:
                # Add image message to chat
                self.add_image_message(image_path)
                
                # Analyze the image
                self.analyze_image(image_path)
    
    def analyze_image(self, image_path):
        """Analyze an image using AI"""
        # Add a system message indicating analysis is in progress
        system_msg = "Analyzing image..."
        self.add_system_message(system_msg)
        
        # If using online API and it's available, use it for image analysis
        if self.chatbot.use_online_api and self.chatbot.online_api.available:
            # For now, just add a placeholder response
            response = "I'm analyzing this image using online AI. This feature would use the API's image analysis capabilities."
            self.add_ai_message(response)
            return
        
        # If using local Ollama, we need to describe the image first
        try:
            # Use a basic image description approach
            import cv2
            import numpy as np
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                self.add_system_message("Failed to load image", is_error=True)
                return
                
            # Get basic image info
            height, width, channels = img.shape
            
            # Create a basic description
            description = f"This is an image of size {width}x{height} pixels. "
            
            # Analyze colors
            avg_color = np.mean(img, axis=(0, 1))
            b, g, r = avg_color  # OpenCV uses BGR
            
            # Determine dominant color
            if r > g and r > b:
                dominant = "red"
            elif g > r and g > b:
                dominant = "green"
            elif b > r and b > g:
                dominant = "blue"
            else:
                dominant = "grayscale"
                
            description += f"The dominant color appears to be {dominant}. "
            
            # Check if image is dark or bright
            brightness = np.mean(img)
            if brightness < 85:
                description += "The image is relatively dark. "
            elif brightness > 170:
                description += "The image is relatively bright. "
            else:
                description += "The image has moderate brightness. "
                
            # Generate a prompt for the AI
            prompt = f"I'm looking at an image with the following basic properties: {description}. Can you provide some insights or ask me questions about what might be in this image?"
            
            # Get AI response
            response = self.chatbot.generate_response(prompt)
            self.add_ai_message(response)
            
        except Exception as e:
            error_msg = f"Error analyzing image: {str(e)}"
            self.add_system_message(error_msg, is_error=True)
    
    def add_user_message(self, content):
        """Add a user message to the chat"""
        # Create and add message bubble
        message_bubble = MessageBubble(content, is_user=True)
        self.chat_layout.addWidget(message_bubble)
        
        # Save to history
        self.chat_history_manager.add_message("user", content)
        
        # Scroll to bottom
        QApplication.processEvents()  # Force update to get correct scroll height
        scroll_area = self.chat_container.parent()
        if hasattr(scroll_area, 'verticalScrollBar'):
            scroll_area.verticalScrollBar().setValue(
                scroll_area.verticalScrollBar().maximum()
            )
    
    def add_ai_message(self, content):
        """Add an AI message to the chat"""
        # Create and add message bubble
        message_bubble = MessageBubble(content, is_user=False)
        self.chat_layout.addWidget(message_bubble)
        
        # Save to history
        self.chat_history_manager.add_message("ai", content)
        
        # Scroll to bottom
        QApplication.processEvents()  # Force update to get correct scroll height
        scroll_area = self.chat_container.parent()
        if hasattr(scroll_area, 'verticalScrollBar'):
            scroll_area.verticalScrollBar().setValue(
                scroll_area.verticalScrollBar().maximum()
            )
    
    def add_system_message(self, content, is_error=False):
        """Add a system message to the chat"""
        # Create and add message bubble
        message_bubble = SystemMessageBubble(content, is_error)
        self.chat_layout.addWidget(message_bubble)
        
        # Save to history
        self.chat_history_manager.add_message("system", content)
        
        # Scroll to bottom
        QApplication.processEvents()  # Force update to get correct scroll height
        scroll_area = self.chat_container.parent()
        if hasattr(scroll_area, 'verticalScrollBar'):
            scroll_area.verticalScrollBar().setValue(
                scroll_area.verticalScrollBar().maximum()
            )
    
    def add_image_message(self, image_path, caption=""):
        """Add an image message to the chat"""
        # Create and add image bubble
        image_bubble = ImageMessageBubble(image_path, caption, is_user=True)
        self.chat_layout.addWidget(image_bubble)
        
        # Save to history
        self.chat_history_manager.add_message("user", caption if caption else "[Image]", image_path)
        
        # Scroll to bottom
        QApplication.processEvents()  # Force update to get correct scroll height
        scroll_area = self.chat_container.parent()
        if hasattr(scroll_area, 'verticalScrollBar'):
            scroll_area.verticalScrollBar().setValue(
                scroll_area.verticalScrollBar().maximum()
            )
    
    def display_chat_history(self):
        """Display the chat history"""
        history = self.chat_history_manager.get_history()
        
        if not history:
            return
            
        # Clear existing widgets
        while self.chat_layout.count():
            item = self.chat_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add system message indicating history is loading
        system_bubble = SystemMessageBubble("Loading previous chat history...")
        self.chat_layout.addWidget(system_bubble)
        
        # Add messages from history
        for message in history:
            role = message.get("role")
            content = message.get("content")
            image_path = message.get("image_path")
            
            if role == "user":
                if image_path:
                    # This is an image message
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
        self.chat_history_manager.clear_history()
        
        # Clear existing widgets
        while self.chat_layout.count():
            item = self.chat_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add system message
        self.add_system_message("Chat history cleared.")
    
    def send_message(self):
        """Send a message to the AI assistant"""
        message = self.chat_input.text().strip()
        if not message:
            return
            
        # Add user message to chat
        self.add_user_message(message)
        
        # Check if Ollama is in offline mode
        if self.chatbot.offline_mode and not message.lower().startswith("restart"):
            # Add status message
            system_msg = "AI is in offline mode. Type 'restart ai' to attempt reconnection."
            self.add_system_message(system_msg, is_error=True)
            
        # If user asks to restart AI, try to reconnect
        if message.lower() in ["restart ai", "restart", "reconnect"]:
            system_msg = "Attempting to reconnect to Ollama service..."
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
            # Get response from AI
            response = self.chatbot.generate_response(message)
            
            # Add AI response to chat
            self.add_ai_message(response)
        
        # Clear input
        self.chat_input.clear()
        
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
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = ChatMateApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
