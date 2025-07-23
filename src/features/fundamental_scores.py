"""C\u00e1lculo de puntajes fundamentales de acciones."""


def score_company(fundamentals):
    """Crea un puntaje simple a partir de datos fundamentales."""
    pe = fundamentals.get("pe_ratio", 0)
    eps = fundamentals.get("eps", 0)
    if pe == 0:
        return 0
    return eps / pe
