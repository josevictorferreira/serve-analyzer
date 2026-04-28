{
  description = "Serve analyzer development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      mkDevShell =
        system: pkgs:
        let
          pythonEnv = pkgs.python3.withPackages (
            ps: with ps; [
              jupyter
              notebook
              jupyterlab
              numpy
              scipy
              matplotlib
              pandas
              opencv4
              pillow
              scikit-image
              ipykernel
              ipympl
              fastapi
              uvicorn
              python-multipart
            ]
          );
        in
        pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.ffmpeg
            pkgs.nodejs
            pkgs.gnumake
            pkgs.zlib
            pkgs.stdenv.cc.cc.lib
          ];
          shellHook = ''
            export PYTHONPATH="$PWD:$PYTHONPATH"
            export JUPYTER_PATH="${pkgs.python3Packages.notebook}/share/jupyter:${pkgs.python3Packages.jupyterlab}/share/jupyter"
            export LD_LIBRARY_PATH="${pkgs.zlib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
            # Create venv for pip packages not in nixpkgs
            if [ ! -d .venv ]; then
              echo "Creating .venv for pip packages (ultralytics, roboflow)..."
              python -m venv .venv --system-site-packages
            fi
            source .venv/bin/activate

            # Install ultralytics + roboflow if not present
            if ! python -c "import ultralytics" 2>/dev/null; then
              echo "Installing ultralytics (without opencv-python)..."
              pip install --quiet ultralytics --no-deps
              pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
              pip install --quiet py-cpuinfo psutil pyyaml tqdm requests pandas seaborn
            fi
            # Install huggingface_hub for RJTPP model download
            if ! python -c "import huggingface_hub" 2>/dev/null; then
              echo "Installing huggingface_hub..."
              pip install --quiet huggingface_hub
            fi
            if ! python -c "import roboflow" 2>/dev/null; then
              echo "Installing roboflow..."
              pip install --quiet roboflow
            fi
            echo ""
            echo "Serve Analyzer dev shell"
            echo "  make dev               # start browser annotation with fine-tuned model if available"
            echo "  make categorize        # same as make dev"
            echo "  make help              # show available project commands"
            echo ""
            echo "The default pre-label model is:"
            echo "  annotation_exports/4f6e0258d09c/training-runs/rjtpp-finetune-1280/weights/best.pt"
            echo ""
            echo "Override pre-label model with:"
            echo "  SERVE_ANALYZER_RJTPP_MODEL_PATH=/path/to/best.pt make categorize"
            echo "  SERVE_ANALYZER_ALLOW_REMOTE_MODEL_DOWNLOAD=1 make categorize"
            echo ""
          '';
        };
    in
    {
      devShells.x86_64-linux.default = mkDevShell "x86_64-linux" (
        import nixpkgs { system = "x86_64-linux"; }
      );
      devShells.aarch64-darwin.default = mkDevShell "aarch64-darwin" (
        import nixpkgs { system = "aarch64-darwin"; }
      );
    };
}
