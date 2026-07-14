"""The provider adapters: one module per API, each one owning its own wire shapes.

A provider decides what turn k sends and echoes back what the server said; core
decides what a turn costs. Nothing else in the suite may know that Gemini spells a
model turn `parts` and OpenAI spells it something else -- that knowledge lives here
and nowhere else, or every chart and every metric grows a provider special case.
"""
