# Pre-merge gate for the Gemini traffic experiment.
.PHONY: preflight test build smoke clean

# Full gate: unit tests + docker build + mock-container smoke. Run before merging.
preflight:
	./preflight.sh

# Unit tests only (fast, no docker, no API key -- the suite runs in mock mode).
test:
	python3 -m venv .venv-preflight
	./.venv-preflight/bin/pip -q install flask requests urllib3 pytest
	./.venv-preflight/bin/python -m pytest tests -q

# Build the image only.
build:
	docker build -t gemini-preflight .

clean:
	rm -rf .venv-preflight
	-docker rmi gemini-preflight 2>/dev/null
