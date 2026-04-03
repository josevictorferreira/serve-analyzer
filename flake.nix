{
  description = "Serve analyzer development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      # Platform: darwin (macOS) only for this project
      system = "aarch64-darwin";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        # Build-time dependencies
        buildInputs = [
          # Python interpreter
          pkgs.python3

          # Jupyter and notebook
          pkgs.python3Packages.jupyter
          pkgs.python3Packages.notebook

          # Numerical and scientific computing
          pkgs.python3Packages.numpy
          pkgs.python3Packages.scipy
          pkgs.python3Packages.matplotlib
          pkgs.python3Packages.pandas

          # Image and video processing
          pkgs.python3Packages.opencv4
          pkgs.python3Packages.pillow
          pkgs.python3Packages.scikit-image

          # Video loading (ffmpeg used by imageio/cv2)
          pkgs.ffmpeg_6

          # Jupyter kernel for notebook
          pkgs.python3Packages.ipykernel
        ];

        # Shell hook to ensure Jupyter can find the kernel
        shellHook = ''
          export PYTHONPATH="$PWD:$PYTHONPATH"
        '';
      };
    };
}
