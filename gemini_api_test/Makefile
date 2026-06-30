# Pre-merge gate for the Gemini traffic experiment.
.PHONY: preflight test build smoke clean

# Full gate: unit tests + docker build + mock-container smoke. Run before merging.
preflight:
	./preflight.sh

# Unit tests only (fast, no docker).
test:
	python3 -m venv .venv-preflight
	./.venv-preflight/bin/pip -q install flask requests urllib3
	./.venv-preflight/bin/python -m unittest discover -s tests

# Build the image only.
build:
	docker build -t gemini-preflight .

clean:
	rm -rf .venv-preflight
	-docker rmi gemini-preflight 2>/dev/null
