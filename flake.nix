{
  description = "Serve analyzer development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      mkDevShell = system: pkgs:
        pkgs.mkShell {
          buildInputs = with pkgs; [
            python3
            python3Packages.jupyter
            python3Packages.notebook
            python3Packages.numpy
            python3Packages.scipy
            python3Packages.matplotlib
            python3Packages.pandas
            python3Packages.opencv4
            python3Packages.pillow
            python3Packages.scikit-image
            ffmpeg
            python3Packages.ipykernel
          ];

          shellHook = ''
            export PYTHONPATH="$PWD:$PYTHONPATH"
          '';
        };
    in
    {
      devShells.x86_64-linux.default = mkDevShell "x86_64-linux" (import nixpkgs { system = "x86_64-linux"; });
      devShells.aarch64-darwin.default = mkDevShell "aarch64-darwin" (import nixpkgs { system = "aarch64-darwin"; });
    };
}
