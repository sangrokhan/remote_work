"""Provider-neutral measurement core.

Everything here is about *how a turn is measured*, never about what a turn says.
Bytes come off the socket, the marks come off the stream, and the record is the
shape both providers agree to produce. A provider adapter that wants to add a
metric adds it here or not at all -- two adapters each measuring in their own way
would produce two number systems that cannot be charted together, which is the
one failure this package exists to prevent.
"""
