# Packaging and the pre-merge gate for the traffic lab.
# Nothing in this file may reach a paid API: every target that runs code runs it
# with TRAFFIC_MOCK=1.
.PHONY: venv cwnd test run docker smoke preflight clean

VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
IMAGE   := token-traffic
CONTAINER := token-traffic-smoke
MOCK    := TRAFFIC_MOCK=1 GEMINI_MOCK=1 OPENAI_MOCK=1

# The virtualenv. Recreated only when requirements.txt is newer than the stamp.
venv: $(VENV)/.stamp
$(VENV)/.stamp: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) -q install -r requirements.txt pytest
	touch $@

# The netlink helper that reads congestion state. Built here as well as on demand by
# core.cwnd, so a checkout can be prepared without starting a run -- and so a compiler
# error shows up in the build rather than three seconds before an arm.
cwnd: native/cwnd_monitor
native/cwnd_monitor: native/cwnd_monitor.c
	cc -O2 -Wall -o $@ $<

# The suite is offline: no API key, no network, no docker. It runs in mock mode.
#
# The collection guard is not paranoia. The previous lab's gate ran
# `unittest discover`, which collects only what subclasses TestCase, quietly
# skipped the ~230 pytest-function tests, reported "1 test OK", and passed --
# a green gate that tested nothing is worse than no gate. So: count what pytest
# collects, and refuse to proceed on zero.
test: venv cwnd
	@n=$$($(PY) -m pytest tests -q --collect-only 2>/dev/null | grep -c '::' || true); \
	if [ "$$n" -lt 1 ]; then \
	  echo "FAIL: pytest collected $$n tests from tests/ -- collection is broken."; \
	  exit 1; \
	fi; \
	echo "collected $$n tests"
	$(MOCK) $(PY) -m pytest tests -q

# The web UI, mocked. http://127.0.0.1:8080
run: venv
	$(MOCK) $(PY) cli.py --serve

docker:
	docker build -t $(IMAGE) .

# The image, actually started, actually answering. Build -> run mocked ->
# /api/config -> a small POST /api/run -> assert the JSON -> kill it.
smoke: docker
	./preflight.sh --smoke-only

# Full gate: tests + build + container smoke.
preflight:
	./preflight.sh

clean:
	rm -f native/cwnd_monitor
	rm -rf $(VENV) .pytest_cache tests/__pycache__ core/__pycache__ providers/__pycache__
	-docker rm -f $(CONTAINER) 2>/dev/null
	-docker rmi $(IMAGE) 2>/dev/null
