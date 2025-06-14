import sys
import os
import re
import traceback
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt6.QtCore import pyqtSlot, QThread, pyqtSignal
from projectx_ui import Ui_MainWindow
from services.llm_generator import LLMGenerator
from services.validator import NixValidator
import nix_merger


class LLMWorker(QThread):
    """Worker thread for LLM generation to prevent UI freezing"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt
        self.llm_generator = LLMGenerator()
    
    def run(self):
        try:
            result = self.llm_generator.generate_nix_config(self.prompt)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Initialize components
        self.validator = NixValidator()
        self.llm_worker = None
        self.config_file_path = "configuration.nix"
        
        # Connect buttons to methods
        self.ui.generateButton.clicked.connect(self.on_generate_clicked)
        self.ui.apply_button.clicked.connect(self.on_apply_clicked)
        self.ui.cancel_button.clicked.connect(self.on_cancel_clicked)
        
        # Monitor changes to new config text
        self.ui.newConfigText.textChanged.connect(self.on_config_changed)
        
        # Initial setup
        self.load_current_config()
        self.update_button_states()
    
    def load_current_config(self):
        """Load the current configuration.nix file and display it"""
        try:
            if os.path.exists(self.config_file_path):
                with open(self.config_file_path, 'r', encoding='utf-8') as f:
                    current_config = f.read()
                self.ui.oldConfigText.setPlainText(current_config)
                print(f"✅ Loaded configuration from {self.config_file_path}")
            else:
                # Create a basic configuration.nix if it doesn't exist
                basic_config = """{ config, pkgs, ... }:

{
  # Basic NixOS configuration
  # Edit this file to customize your system

  # Bootloader
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;

  # Enable networking
  networking.networkmanager.enable = true;

  # Set your time zone
  time.timeZone = "America/New_York";

  # Select internationalisation properties
  i18n.defaultLocale = "en_US.UTF-8";

  # Define a user account
  users.users.user = {
    isNormalUser = true;
    description = "user";
    extraGroups = [ "networkmanager" "wheel" ];
  };

  # Enable the X11 windowing system
  services.xserver.enable = true;
  services.xserver.displayManager.gdm.enable = true;
  services.xserver.desktopManager.gnome.enable = true;

  # Allow unfree packages
  nixpkgs.config.allowUnfree = true;

  # List packages installed in system profile
  environment.systemPackages = with pkgs; [
    vim
    wget
    git
  ];

  # This value determines the NixOS release from which the default
  # settings for stateful data, like file locations and database versions
  # on your system were taken. It's perfectly fine and recommended to leave
  # this value at the release version of the first install of this system.
  system.stateVersion = "23.11";
}"""
                with open(self.config_file_path, 'w', encoding='utf-8') as f:
                    f.write(basic_config)
                self.ui.oldConfigText.setPlainText(basic_config)
                print(f"✅ Created basic configuration at {self.config_file_path}")
        except Exception as e:
            error_msg = f"Failed to load configuration.nix: {str(e)}"
            print(f"❌ {error_msg}")
            self.show_error_message("Configuration Load Error", error_msg)
            # Set empty text if loading fails
            self.ui.oldConfigText.setPlainText("")
    
    def extract_nix_code(self, llm_output):
        """Extract pure Nix code from LLM output, removing explanations"""
        # The LLM output should already be cleaned, but let's ensure it's properly formatted
        cleaned = llm_output.strip()
        
        # If the output doesn't start with {, it might be a simple attribute
        # In that case, we should wrap it or use as-is depending on the content
        if not cleaned.startswith('{') and not cleaned.startswith('#'):
            # This might be a simple attribute like "programs.fontconfig = { enable = true; };"
            # Let's check if it's a complete statement
            if '=' in cleaned and cleaned.endswith(';'):
                return cleaned
            elif '=' in cleaned and not cleaned.endswith(';'):
                return cleaned + ';'
        
        # For block configurations, ensure proper formatting
        lines = cleaned.split('\n')
        
        # Remove empty lines at the beginning and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        
        if not lines:
            return cleaned
        
        # If it's a block configuration, ensure it's properly formatted
        result = '\n'.join(lines)
        
        return result
    
    @pyqtSlot()
    def on_generate_clicked(self):
        prompt = self.ui.promptInput.toPlainText().strip()
        if not prompt:
            self.ui.llmOutput.setPlainText("Prompt is empty. Please enter a prompt.")
            return
        
        # Disable generate button during processing
        self.ui.generateButton.setEnabled(False)
        self.ui.generateButton.setText("Generating...")
        self.ui.llmOutput.setPlainText("Generating configuration...")
        
        # Start LLM generation in worker thread
        self.llm_worker = LLMWorker(prompt)
        self.llm_worker.finished.connect(self.on_llm_generation_finished)
        self.llm_worker.error.connect(self.on_llm_generation_error)
        self.llm_worker.start()
    
    @pyqtSlot(str)
    def on_llm_generation_finished(self, generated_config):
        """Handle successful LLM generation"""
        try:
            # Extract clean Nix code
            clean_nix_code = self.extract_nix_code(generated_config)
            self.ui.llmOutput.setPlainText(clean_nix_code)
            
            # Get current configuration
            current_config = self.ui.oldConfigText.toPlainText()
            
            # Merge configurations
            try:
                merged_config = nix_merger.merge_configs_python(current_config, clean_nix_code)
                
                # Validate the merged configuration
                # Write to temporary file for validation
                temp_file = "temp_config.nix"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(merged_config)
                self.ui.newConfigText.setPlainText(merged_config)
                """
                if self.validator.validate_config(temp_file):
                    self.ui.newConfigText.setPlainText(merged_config)
                    # Clean up temp file
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                else:
                    # Show validation error
                    error_msg = self.validator.get_last_error()
                    self.show_error_message("Validation Error", 
                                           f"Generated configuration is invalid:\n\n{error_msg}")
                    # Clean up temp file
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                """    
            except Exception as e:
                self.show_error_message("Merge Error", f"Failed to merge configurations:\n\n{str(e)}")
                
        except Exception as e:
            self.show_error_message("Processing Error", f"Failed to process generated configuration:\n\n{str(e)}")
        
        finally:
            # Re-enable generate button
            self.ui.generateButton.setEnabled(True)
            self.ui.generateButton.setText("Generate")
            self.update_button_states()
    
    @pyqtSlot(str)
    def on_llm_generation_error(self, error_message):
        """Handle LLM generation error"""
        self.show_error_message("Generation Error", f"Failed to generate configuration:\n\n{error_message}")
        self.ui.llmOutput.setPlainText(f"Error: {error_message}")
        
        # Re-enable generate button
        self.ui.generateButton.setEnabled(True)
        self.ui.generateButton.setText("Generate")
        self.update_button_states()
    
    @pyqtSlot()
    def on_apply_clicked(self):
        """Apply the new configuration by saving it to configuration.nix"""
        try:
            new_config = self.ui.newConfigText.toPlainText()
            
            # Create backup of current configuration
            backup_path = f"{self.config_file_path}.backup"
            if os.path.exists(self.config_file_path):
                with open(self.config_file_path, 'r', encoding='utf-8') as f:
                    backup_content = f.read()
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(backup_content)
            
            # Save new configuration
            with open(self.config_file_path, 'w', encoding='utf-8') as f:
                f.write(new_config)
            
            # Update old config text to reflect the new current state
            self.ui.oldConfigText.setPlainText(new_config)
            
            # Show success message
            QMessageBox.information(self, "Success", 
                                  f"Configuration saved to {self.config_file_path}\n"
                                  f"Backup created: {backup_path}")
            
            self.update_button_states()
            
        except Exception as e:
            self.show_error_message("Save Error", f"Failed to save configuration:\n\n{str(e)}")
    
    @pyqtSlot()
    def on_cancel_clicked(self):
        """Cancel changes and clear the new configuration"""
        self.ui.newConfigText.clear()
        self.ui.llmOutput.clear()
        self.update_button_states()
    
    @pyqtSlot()
    def on_config_changed(self):
        """Handle changes to the new configuration text"""
        self.update_button_states()
    
    def update_button_states(self):
        """Update the enabled state of buttons based on current state"""
        old_config = self.ui.oldConfigText.toPlainText().strip()
        new_config = self.ui.newConfigText.toPlainText().strip()
        
        # Enable apply/cancel buttons only if there's new config and it's different from old
        enable = bool(new_config) and (new_config != old_config)
        self.ui.apply_button.setEnabled(enable)
        self.ui.cancel_button.setEnabled(enable)
    
    def show_error_message(self, title, message):
        """Show an error message box"""
        QMessageBox.critical(self, title, message)
    
    def closeEvent(self, event):
        """Handle application close event"""
        # Clean up worker thread if running
        if self.llm_worker and self.llm_worker.isRunning():
            self.llm_worker.terminate()
            self.llm_worker.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
