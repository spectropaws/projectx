import subprocess
import os

def check_nixos_config_strict(nix_file_path):
    abs_path = os.path.abspath(nix_file_path)
    try:
        result = subprocess.run(
            [
                "nix", "eval", "--impure",
                "--expr", f'''
                  let
                    nixpkgs = import <nixpkgs> {{}};
                    config = import (nixpkgs.path + "/nixos/lib/eval-config.nix") {{
                      system = "x86_64-linux";
                      modules = [ "{abs_path}" ];
                    }};
                  in config.config.system.build.toplevel
                ''',
                "--show-trace"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            print("❌ Configuration is invalid:")
            print(result.stderr)
            return False

        print("✅ Configuration is valid and type-checked.")
        return True

    except FileNotFoundError:
        print("❌ `nix` command not found.")
        return False

check_nixos_config_strict("new.nix")

