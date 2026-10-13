def _f(x, nd=0):
    """Zahl (oder Array) für den LaTeX-Mathmodus formatieren, Dünnraum als Tausendertrenner."""
    if np.ndim(x) > 0:
        return "[" + ", ".join(_f(v, nd) for v in np.ravel(x)) + "]"
    return f"{x:,.{nd}f}".replace(",", r"\,")


def _e(x, nd=3):
    """Wissenschaftliche Notation für LaTeX."""
    m, ex = f"{x:.{nd}e}".split("e")
    return rf"{m} \cdot 10^{{{int(ex)}}}"


def _ok(cond) -> str:
    return "✔" if cond else "✘"
