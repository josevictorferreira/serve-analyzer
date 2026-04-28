FINE_TUNED_MODEL := $(CURDIR)/annotation_exports/4f6e0258d09c/training-runs/rjtpp-finetune-1280/weights/best.pt

.PHONY: help analyze-video categorize dev

help:
	@printf '%s\n' \
		'Available commands:' \
		'  make dev               Start browser annotation using fine-tuned model if available' \
		'  make categorize        Start backend + frontend for ball annotation' \
		'  make analyze-video     Run multi-serve analysis on video.mov' \
		'' \
		'Default categorization model:' \
		'  annotation_exports/4f6e0258d09c/training-runs/rjtpp-finetune-1280/weights/best.pt' \
		'' \
		'RJTPP pre-label options for categorization:' \
		'  SERVE_ANALYZER_RJTPP_MODEL_PATH=/path/to/best.pt make categorize' \
		'  SERVE_ANALYZER_ALLOW_REMOTE_MODEL_DOWNLOAD=1 make categorize'

analyze-video:
	nix develop --command bash -lc 'if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi; python -m serve_analyzer.multi_serve video.mov --model yolov8n.pt --frame-skip 4 -o serves_analysis.json'

dev: categorize

categorize:
	@bash -lc 'set -euo pipefail; \
		if [ -z "$${IN_NIX_SHELL:-}" ]; then \
			echo "Run this inside nix develop, or use: nix develop --command make dev"; \
			exit 1; \
		fi; \
		default_model="$(FINE_TUNED_MODEL)"; \
		if [ -n "$${SERVE_ANALYZER_RJTPP_MODEL_PATH:-}" ]; then \
			if [ ! -f "$${SERVE_ANALYZER_RJTPP_MODEL_PATH}" ]; then \
				echo "Configured pre-label model does not exist: $${SERVE_ANALYZER_RJTPP_MODEL_PATH}"; \
				exit 1; \
			fi; \
			echo "Using configured pre-label model: $${SERVE_ANALYZER_RJTPP_MODEL_PATH}"; \
		elif [ -f "$$default_model" ]; then \
			export SERVE_ANALYZER_RJTPP_MODEL_PATH="$$default_model"; \
			echo "Using fine-tuned pre-label model: $$default_model"; \
		else \
			echo "Fine-tuned model not found: $$default_model"; \
			echo "Start with no pre-label model, or set SERVE_ANALYZER_RJTPP_MODEL_PATH."; \
		fi; \
		echo "Starting annotation backend on http://127.0.0.1:8000"; \
		python -m web.backend & backend_pid=$$!; \
		echo "Starting annotation UI on http://localhost:5173"; \
		(cd web && npm run dev) & frontend_pid=$$!; \
		trap "kill $$backend_pid $$frontend_pid 2>/dev/null || true" INT TERM EXIT; \
		wait -n $$backend_pid $$frontend_pid; \
		status=$$?; \
		kill $$backend_pid $$frontend_pid 2>/dev/null || true; \
		exit $$status'
	echo manual | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level
	sudo rocm-smi --setpoweroverdrive 283
	sudo rocm-smi --setsrange 500 2400
	sudo rocm-smi --showclocks
	sudo rocm-smi --showpower
gpu-cap:
