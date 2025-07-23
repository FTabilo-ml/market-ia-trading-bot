"""An\u00e1lisis de sentimiento b\u00e1sico sobre titulares."""

from textblob import TextBlob


def analyze_headlines(headlines):
    """Devuelve el puntaje promedio de sentimiento."""
    if not headlines:
        return 0.0
    scores = [TextBlob(h).sentiment.polarity for h in headlines]
    return sum(scores) / len(scores)
