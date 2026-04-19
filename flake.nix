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
            ]
          );
        in
        pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.ffmpeg
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
