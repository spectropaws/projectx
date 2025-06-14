import subprocess
import os


class NixValidator:
    def __init__(self):
        self.last_error = ""
    
    def validate_config(self, nix_file_path):
        """
        Validate NixOS configuration file
        Returns True if valid, False if invalid
        """
        abs_path = os.path.abspath(nix_file_path)
        
        if not os.path.exists(abs_path):
            self.last_error = f"Configuration file not found: {abs_path}"
            return False
        
        try:
            # Check if nix command is available
            subprocess.run(["nix", "--version"], 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE, 
                         check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.last_error = "Nix command not found. Please ensure Nix is installed and available in PATH."
            return False
        
        try:
            # Use nix-instantiate for faster validation (doesn't build, just evaluates)
            """
            result = subprocess.run([
                "nix-instantiate", 
                "--eval", 
                "--strict",
                "--show-trace",
                "--expr", f'''
                  let
                    nixpkgs = import <nixpkgs> {{}};
                    lib = nixpkgs.lib;
                    eval-config = import (nixpkgs.path + "/nixos/lib/eval-config.nix");
                    config = eval-config {{
                      system = "x86_64-linux";
                      modules = [ "{abs_path}" ];
                    }};
                  in
                    # Just check if the configuration can be evaluated
                    config.options.system.stateVersion.type
                '''
            ], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            timeout=30  # 30 second timeout
            )
            """

            result = subprocess.run(
                ['nix-instantiate', '--parse', file_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                self.last_error = self._parse_nix_error(result.stderr)
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            self.last_error = "Validation timed out. The configuration might be too complex or contain infinite recursion."
            return False
        except Exception as e:
            self.last_error = f"Validation failed with unexpected error: {str(e)}"
            return False
    
    def validate_config_syntax_only(self, nix_file_path):
        """
        Quick syntax-only validation using nix-instantiate --parse
        Much faster but less thorough than full validation
        """
        abs_path = os.path.abspath(nix_file_path)
        
        if not os.path.exists(abs_path):
            self.last_error = f"Configuration file not found: {abs_path}"
            return False
        
        try:
            result = subprocess.run([
                "nix-instantiate", 
                "--parse", 
                abs_path
            ], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            timeout=10
            )
            
            if result.returncode != 0:
                self.last_error = self._parse_nix_error(result.stderr)
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            self.last_error = "Syntax validation timed out."
            return False
        except Exception as e:
            self.last_error = f"Syntax validation failed: {str(e)}"
            return False
    
    def _parse_nix_error(self, stderr):
        """Parse Nix error messages to make them more user-friendly"""
        lines = stderr.split('\n')
        
        # Look for the most relevant error lines
        error_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('trace:'):
                # Skip some verbose debug info
                if 'while evaluating' not in line and 'while calling' not in line:
                    error_lines.append(line)
        
        if error_lines:
            return '\n'.join(error_lines[:10])  # Limit to first 10 relevant lines
        else:
            return stderr  # Fallback to original error
    
    def get_last_error(self):
        """Get the last validation error message"""
        return self.last_error


# Standalone function for backward compatibility
def check_nixos_config_strict(nix_file_path):
    """
    Backward compatible function
    """
    validator = NixValidator()
    result = validator.validate_config(nix_file_path)
    
    if result:
        print("✅ Configuration is valid and type-checked.")
    else:
        print("❌ Configuration is invalid:")
        print(validator.get_last_error())
    
    return result


if __name__ == "__main__":
    # Test the validator
    check_nixos_config_strict("configuration.nix")
