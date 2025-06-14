{ pkgs ? import <nixpkgs> {} }:
let
  nixMerger = pkgs.callPackage ./nix-merger.nix {};
in
pkgs.mkShell {
  packages = with pkgs; [
    python313
    python313Packages.pip
    python313Packages.setuptools
    python313Packages.llama-cpp-python
    python313Packages.langchain
    python313Packages.langchain-community
    python313Packages.pydantic
    python313Packages.pydantic-core
    
    python313Packages.numpy
    python313Packages.sentence-transformers
    python313Packages.faiss
    
    python313Packages.pyqt6
    nix-eval-jobs
    nixMerger  # Your custom package
  ];
  shellHook = ''
    echo "✅ Shell ready with nix-merger available."
  '';
}
