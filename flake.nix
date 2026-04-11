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
          ];

          shellHook = ''
            export PYTHONPATH="$PWD:$PYTHONPATH"
            export JUPYTER_PATH="${pkgs.python3Packages.notebook}/share/jupyter:${pkgs.python3Packages.jupyterlab}/share/jupyter"

            # Create venv for pip packages not in nixpkgs (e.g., ultralytics)
            if [ ! -d .venv ]; then
              echo "Creating .venv for pip packages (ultralytics)..."
              python -m venv .venv --system-site-packages
            fi
            source .venv/bin/activate

            # Install ultralytics if not present (without opencv since nix provides it)
            if ! python -c "import ultralytics" 2>/dev/null; then
              echo "Installing ultralytics (without opencv-python)..."
              pip install --quiet ultralytics --no-deps
              pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
              pip install --quiet py-cpuinfo psutil pyyaml tqdm requests pandas seaborn
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
