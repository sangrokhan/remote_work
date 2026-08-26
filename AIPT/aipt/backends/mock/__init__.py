"""aipt.backends.mock -- fixed/replayed JSON I/O backend (DESIGN.md 4.5).

Generalizes the former ``tcp_congestion`` mock server under the
``aipt.backends.base.Backend`` protocol (DESIGN.md 5, A3/B1/B3):

  * ``server.py``       -- migrated HTTP/1.1 keep-alive mock server,
                            extended to serve fixture answer text.
  * ``fixtures.py``     -- Q&A JSON fixture loader + byte-size-sweep mode
                            (B1).
  * ``replay.py``       -- turns a captured real exchange into a
                            byte-pattern-only replay fixture (B3).
  * ``conversation.py`` -- migrated multi-turn scripted ``run()``, plus
                            ``MockBackend`` (the ``Backend`` protocol
                            implementation client code actually uses).
  * ``probe.py``        -- migrated idle-window RTT probe.
"""

from __future__ import annotations

from aipt.backends.mock.conversation import MockBackend

NAME = "mock"

__all__ = ["NAME", "MockBackend"]
