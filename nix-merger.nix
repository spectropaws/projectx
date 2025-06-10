{ pkgs }:
pkgs.python313Packages.buildPythonPackage rec {
  pname = "nix-merger";
  version = "0.1.0";
  
  src = ./lib/nix_merger-0.1.0-cp313-cp313-manylinux_2_34_x86_64.whl;
  format = "wheel";
  
  # Add any Python dependencies your PyO3 package needs
  propagatedBuildInputs = with pkgs.python313Packages; [
    # Add dependencies here if needed
  ];
  
  # Skip tests since it's a prebuilt wheel
  doCheck = false;
}
