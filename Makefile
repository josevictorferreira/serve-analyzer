.PHONY: analyze-video

analyze-video:
	nix develop --command bash -lc 'if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi; python -m serve_analyzer.multi_serve video.mov --model yolov8n.pt --frame-skip 4 -o serves_analysis.json'
