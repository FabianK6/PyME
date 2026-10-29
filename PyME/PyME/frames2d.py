#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Ebene Rahmen-, Balken- und Fachwerktragwerke (2D) mit der direkten Steifigkeitsmethode. Nur numpy.

Stab: Euler-Bernoulli-Rahmenelement (Normalkraft + Biegung), konstanter Querschnitt je Stab, beliebige Lage in der
x-y-Ebene, Streckenlasten (konstant oder linear, lokal oder global), Momentengelenke an den Stabenden,
Einzelkräfte/-momente an Knoten, Lager (fest, gelenkig, Rollenlager auch schräg, Parallelführung, Federn),
Stützensenkungen. Statisch bestimmte und unbestimmte Systeme.

Klassen
    Node          Knoten mit Koordinaten, Lagerung und Knotenlasten
    Element       Stab zwischen zwei Knoten (Rahmenstab; Element.truss_bar für Fachwerkstäbe)
    FrameSystem   Tragwerk: solve() und alle Ergebnisse (Verschiebungen, Reaktionen, Schnittgrössen, Diagramme, Bericht)

Einheiten: N, mm, N/mm² (E), mm² (A), mm⁴ (I)  ->  Kräfte in N, Momente in Nmm, Verschiebungen in mm, Winkel in rad.

Koordinaten und Vorzeichen
    global:  x nach rechts, y nach oben; Kräfte positiv in +x / +y; Momente und Drehwinkel positiv gegen den Uhrzeigersinn.
    lokal:   x_l vom Stabanfang zum Stabende, y_l um 90° gegen den Uhrzeigersinn gedreht.
    Schnittgrössen (wie im Balkenskript): N > 0 Zug; Q > 0 wenn die Kraft am rechten Rand des linken Teilstücks nach unten
             wirkt (Q' = q_y); M > 0 wenn die Zugfaser auf der lokalen -y_l-Seite liegt (M' = Q, w'' = M/EI).
             Ein waagerechter Balken von links nach rechts verhält sich damit wie im Balkenskript (Q(0) = Auflagerkraft).

Beispiel 1: Balken
    n1 = Node(1, 0, 0).set_as_pinned()
    n2 = Node(2, 4000, 0).set_as_roller()
    beam = Element(n1, n2, 1, E=210_000, A=2000, I=8e6)
    beam.set_load(qy=-5)                           # N/mm, lokal; nach unten negativ
    system = FrameSystem([beam]).solve()
    display(Markdown(system.markdown()))
    system.plot("M")                               # optional, braucht matplotlib
    (Ein waagerechter Balken braucht ein gelenkiges Lager, sonst bleibt er in x verschieblich.)

Beispiel 2: Fachwerk (alle Stäbe gelenkig, nur Normalkräfte)
    Fünf Knoten, sieben Stäbe, Festlager links (A), Loslager rechts (C). Last: 10 kN nach unten an B, 5 kN nach rechts an D.

              D --- E            D (1000, 1500)   E (3000, 1500)   [mm]
             / \   / \
            /   \ /   \          Stäbe: U1 = A-B, U2 = B-C (Untergurt), O1 = D-E (Obergurt),
           A --- B --- C         D1 = A-D, D2 = D-B, D3 = B-E, D4 = E-C (Diagonalen)
                                 A (0, 0)   B (2000, 0)   C (4000, 0)     (Skizze nicht massstäblich)

    from IPython.display import Markdown, display

    E, A = 210_000, 1_000                          # Stahl [N/mm²], Stabquerschnitt [mm²]

    # Knoten (Name, x, y in mm) und Lager
    a = Node("A", 0, 0).set_as_pinned()            # Festlager
    b = Node("B", 2000, 0)
    c = Node("C", 4000, 0).set_as_roller()         # Loslager (hält in y)
    d = Node("D", 1000, 1500)
    e = Node("E", 3000, 1500)

    # Knotenlasten (global, N)
    b.set_force(fy=-10_000)                        # 10 kN nach unten
    d.set_force(fx=5_000)                          # 5 kN nach rechts

    # Stäbe: truss_bar = beidseitig gelenkig, nimmt nur Normalkraft auf (Biegesteifigkeit ist unwichtig)
    bars = [Element.truss_bar(s, t, name, E, A) for name, s, t in [
        ("U1", a, b), ("U2", b, c),                # Untergurt
        ("O1", d, e),                              # Obergurt
        ("D1", a, d), ("D2", d, b), ("D3", b, e), ("D4", e, c),   # Diagonalen
    ]]

    truss = FrameSystem(bars).solve()

    # Stabkräfte N [N], Zug positiv (ohne Last im Stabfeld ist N am Anfang = N am Ende)
    for bar, f in zip(bars, truss.element_forces()):
        print(bar.subscript, round(f[0], 1))
    # U1 7083.3   U2 4583.3   O1 -9166.7   D1 -3755.8   D2 3755.8   D3 8262.7   D4 -8262.7

    print(truss.reactions())                       # A: Rx = -5000, Ry = 3125 N;  C: Ry = 6875 N
    print(truss.displacements()[1][:2])            # Verschiebung von B (ux, uy) = (0.0675, -0.2192) mm

    display(Markdown(truss.markdown()))            # Bericht mit Tabellen
    truss.plot("N")                                # Normalkraftverlauf (optional, matplotlib)
    truss.plot("deformed")                         # verformte Lage

    Die Knotenverdrehungen sind bei reinen Gelenkstäben unbestimmt; solve() setzt sie auf 0 (truss.auto_fixed).
    Kontrolliert mit dem Knotengleichgewicht von Hand (Abweichung < 1e-9 N).
"""
import numpy as np

_GX, _GW = np.polynomial.legendre.leggauss(5)       # Gauss-Punkte für die äquivalenten Knotenlasten


def _g(x, nd=6):
    """Zahl für Berichte auf nd signifikante Stellen formatieren (ohne unnötige Nullen)."""
    return f"{x:.{nd}g}"


class Node(object):
    """
    Knoten des Tragwerks: Koordinaten, Lagerung und Knotenlasten.

    Jeder Knoten hat drei Freiheitsgrade: Verschiebung u_x und u_y sowie Verdrehung phi_z (gegen den Uhrzeigersinn positiv).
    Lager sind lineare Bedingungen a_x*u_x + a_y*u_y + a_r*phi_z = Wert. Dadurch sind auch schräge Rollenlager und
    Stützensenkungen möglich. Federn wirken zusätzlich auf die Steifigkeitsmatrix. Die set_*-Methoden geben den Knoten
    zurück und lassen sich verketten.

    Attributes:
        subscript: Nummer oder Name des Knotens (nur zur Anzeige)
        x, y (float): Koordinaten [mm]
        force (ndarray): Knotenlast [Fx, Fy, Mz] in N bzw. Nmm (global)
        constraints (list): Lagerbedingungen als Tupel (a_x, a_y, a_r, Wert)
        spring (ndarray): Federsteifigkeiten [k_x, k_y, k_r]
        index (int): Position in FrameSystem.nodes (wird von FrameSystem gesetzt)

    Example:
        a = Node("A", 0, 0).set_as_pinned()
        b = Node("B", 4000, 0).set_as_roller().set_force(fy=-1000)
    """

    def __init__(self, subscript: any, x: float, y: float) -> None:
        """
        Knoten mit Koordinaten.

        Args:
            subscript (str | int): Nummer/Name des Knotens
            x, y (float): Koordinaten [mm]
        """
        self.subscript = subscript
        self.x, self.y = float(x), float(y)
        self.force = np.zeros(3)         # Fx, Fy, Mz (global)
        self.constraints = []            # Zeilen (ax, ay, ar, Wert): ax*ux + ay*uy + ar*rz = Wert
        self.spring = np.zeros(3)        # Federsteifigkeiten kx, ky, kr
        self.index = None

    # ------------------------------------------------------------------ Lager
    def fix(self, ux: float | None = None, uy: float | None = None, rz: float | None = None):
        """Vorgegebene Verschiebung in x, y bzw. Verdrehung (None = frei). Werte ungleich 0 sind Stützensenkungen."""
        for k, val in enumerate((ux, uy, rz)):
            if val is not None:
                row = [0.0, 0.0, 0.0]
                row[k] = 1.0
                self.constraints.append((*row, float(val)))
        return self

    def set_as_fixed(self):
        """Einspannung: ux = uy = rz = 0."""
        self.constraints = []
        return self.fix(0, 0, 0)

    def set_as_pinned(self):
        """Gelenkiges Festlager: ux = uy = 0."""
        self.constraints = []
        return self.fix(ux=0, uy=0)

    def set_as_roller(self, angle: float = 0.0, settlement: float = 0.0):
        """
        Rollenlager. Die Unterlage hat die Richtung angle (Grad gegen die x-Achse); senkrecht dazu ist der Knoten
        gehalten, die Reaktion wirkt senkrecht zur Unterlage. angle = 0: Lager hält in y (Loslager).
        """
        a = np.radians(angle)
        self.constraints = [(-np.sin(a), np.cos(a), 0.0, float(settlement))]
        return self

    def set_as_glider(self, angle: float = 0.0):
        """Parallelführung: Verdrehung gesperrt und senkrecht zur Richtung angle gehalten, in Richtung angle verschieblich."""
        a = np.radians(angle)
        self.constraints = [(-np.sin(a), np.cos(a), 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)]
        return self

    def set_as_loose(self):
        """Freier Knoten: Lager und Federn entfernen."""
        self.constraints = []
        self.spring = np.zeros(3)
        return self

    def set_spring(self, kx: float = 0.0, ky: float = 0.0, kr: float = 0.0):
        """Elastische Lagerung: Federn in x [N/mm], y [N/mm] und Drehfeder [Nmm/rad]."""
        self.spring = np.array([kx, ky, kr], dtype=float)
        return self

    # ------------------------------------------------------------------ Lasten
    def set_force(self, fx: float = 0.0, fy: float = 0.0, mz: float = 0.0):
        """Einzelkraft [N] in globaler x/y-Richtung und Einzelmoment [Nmm] (gegen den Uhrzeigersinn positiv). Ersetzt frühere Werte."""
        self.force = np.array([fx, fy, mz], dtype=float)
        return self


class Element(object):
    """
    Rahmenstab (Euler-Bernoulli) zwischen zwei Knoten mit konstantem Querschnitt.

    Lokale Achsen: x_l zeigt vom Anfangs- zum Endknoten, y_l ist um 90° gegen den Uhrzeigersinn gedreht. Lokale
    Freiheitsgrade in der Reihenfolge [u1, v1, phi1, u2, v2, phi2]. Streckenlasten werden über set_load gesetzt,
    Momentengelenke über hinge_start/hinge_end (statische Kondensation). Für Fachwerkstäbe gibt es Element.truss_bar.

    Attributes:
        start, end (Node): Anfangs- und Endknoten
        subscript: Nummer oder Name des Stabes
        E, A, I (float): Elastizitätsmodul [N/mm²], Fläche [mm²], Flächenträgheitsmoment [mm⁴]
        length (float): Stablänge [mm]
        c, s (float): Richtungskosinus und -sinus von x_l in globalen Koordinaten
        angle (float): Winkel von x_l gegen die globale x-Achse [Grad]
        released (list): lokale Freiheitsgrade mit Gelenk (2 = Anfang, 5 = Ende)
        qx, qy (tuple): lokale Streckenlast (Anfangswert, Endwert) [N/mm]

    Nach FrameSystem.solve() zusätzlich:
        d_local (ndarray): lokale Verschiebungen [u1, v1, phi1, u2, v2, phi2] (Gelenkverdrehungen eingerechnet)
        end_forces (ndarray): Kräfte der Knoten auf den Stab [Fx1, Fy1, M1, Fx2, Fy2, M2], lokal, M gegen den Uhrzeigersinn
    """

    def __init__(self,
        start: Node,
        end: Node,
        subscript: any,
        E: float,
        A: float,
        I: float,
        hinge_start: bool = False,
        hinge_end: bool = False
    ) -> None:
        """
        Rahmenstab zwischen zwei Knoten.

        Args:
            start, end (Node): Anfangs- und Endknoten (lokale x-Achse zeigt von start nach end)
            E (float): Elastizitätsmodul [N/mm²]
            A (float): Querschnittsfläche [mm²]
            I (float): Flächenträgheitsmoment [mm⁴]
            hinge_start, hinge_end (bool): Momentengelenk am Stabanfang bzw. Stabende
        """
        self.start, self.end, self.subscript = start, end, subscript
        self.E, self.A, self.I = float(E), float(A), float(I)
        dx, dy = end.x - start.x, end.y - start.y
        self.length = float(np.hypot(dx, dy))
        if self.length == 0:
            raise ValueError(f"Stab {subscript}: Länge 0 (Anfangs- und Endknoten identisch)")
        self.c, self.s = dx / self.length, dy / self.length
        self.angle = float(np.degrees(np.arctan2(dy, dx)))
        self.released = [i for i, h in ((2, hinge_start), (5, hinge_end)) if h]
        self.qx = (0.0, 0.0)             # lokale Streckenlast in x_l am Anfang/Ende [N/mm]
        self.qy = (0.0, 0.0)             # lokale Streckenlast in y_l am Anfang/Ende [N/mm]

    @classmethod
    def truss_bar(cls, start: Node, end: Node, subscript: any, E: float, A: float):
        """
        Fachwerkstab: an beiden Enden gelenkig angeschlossen, nimmt nur Normalkraft auf.

        Args:
            start, end (Node): Anfangs- und Endknoten
            subscript: Nummer oder Name des Stabes
            E (float): Elastizitätsmodul [N/mm²]
            A (float): Querschnittsfläche [mm²]

        Returns:
            Element: Stab mit Momentengelenken an beiden Enden (I ist ohne Bedeutung und wird auf 1 gesetzt).
            Streckenlasten quer zum Stab (z.B. Eigengewicht) erzeugen dann Biegung wie bei einem Einfeldträger.
        """
        return cls(start, end, subscript, E, A, 1.0, hinge_start=True, hinge_end=True)

    def set_load(self, qx: float = 0.0, qy: float = 0.0,
                 qx_end: float | None = None, qy_end: float | None = None, system: str = "local"):
        """
        Streckenlast [N/mm Stablänge], konstant oder linear von Anfang (qx, qy) bis Ende (qx_end, qy_end). Ersetzt frühere Werte.

        Args:
            system (str): "local" (x_l entlang, y_l quer zum Stab) oder "global" (Komponenten in x und y, z.B. Eigengewicht qy = -0.01)
        """
        qxe = qx if qx_end is None else qx_end
        qye = qy if qy_end is None else qy_end
        if system.lower() == "global":
            c, s = self.c, self.s
            qx, qy, qxe, qye = c * qx + s * qy, -s * qx + c * qy, c * qxe + s * qye, -s * qxe + c * qye
        elif system.lower() != "local":
            raise ValueError("system: 'local' oder 'global'")
        self.qx, self.qy = (float(qx), float(qxe)), (float(qy), float(qye))
        return self

    # ------------------------------------------------------------------ Elementmatrizen (lokal)
    def transformation(self):
        """
        Transformationsmatrix T (6x6) von globalen zu lokalen Freiheitsgraden: d_local = T @ d_global.

        Returns:
            ndarray: Blockdiagonalmatrix aus zwei Drehmatrizen (Drehung um den Stabwinkel, Verdrehung bleibt unverändert)
        """
        R = np.array([[self.c, self.s, 0], [-self.s, self.c, 0], [0, 0, 1]])
        T = np.zeros((6, 6))
        T[:3, :3], T[3:, 3:] = R, R
        return T

    def local_stiffness(self):
        """
        Steifigkeitsmatrix des Stabes im lokalen System (6x6), ohne Gelenke.

        Reihenfolge der Freiheitsgrade [u1, v1, phi1, u2, v2, phi2]; Normalkraft über EA/L, Biegung über EI (Euler-Bernoulli).

        Returns:
            ndarray: symmetrische Matrix [N/mm, N/rad, Nmm/rad]
        """
        L, EA, EI = self.length, self.E * self.A, self.E * self.I
        a, b, c, d = EA / L, 12 * EI / L ** 3, 6 * EI / L ** 2, EI / L
        return np.array([
            [a, 0, 0, -a, 0, 0],
            [0, b, c, 0, -b, c],
            [0, c, 4 * d, 0, -c, 2 * d],
            [-a, 0, 0, a, 0, 0],
            [0, -b, -c, 0, b, -c],
            [0, c, 2 * d, 0, -c, 4 * d]])

    def local_equivalent_loads(self):
        """Äquivalente Knotenlasten der Streckenlast (lokal), konsistent mit den Hermite-Ansätzen (exakt)."""
        L = self.length
        x = (_GX + 1) / 2 * L
        w = _GW * L / 2
        xi = x / L
        qx = self.qx[0] + (self.qx[1] - self.qx[0]) * xi
        qy = self.qy[0] + (self.qy[1] - self.qy[0]) * xi
        N1 = 1 - 3 * xi ** 2 + 2 * xi ** 3
        N2 = L * (xi - 2 * xi ** 2 + xi ** 3)
        N3 = 3 * xi ** 2 - 2 * xi ** 3
        N4 = L * (-xi ** 2 + xi ** 3)
        return np.array([np.sum(w * (1 - xi) * qx), np.sum(w * N1 * qy), np.sum(w * N2 * qy),
                         np.sum(w * xi * qx), np.sum(w * N3 * qy), np.sum(w * N4 * qy)])

    def _condensed(self):
        """Steifigkeit und Lastvektor nach statischer Kondensation der Gelenke (6x6, Gelenkzeilen = 0)."""
        k, f = self.local_stiffness(), self.local_equivalent_loads()
        r = self.released
        if not r:
            return k, f
        a = [i for i in range(6) if i not in r]
        krr = k[np.ix_(r, r)]
        kc = np.zeros((6, 6))
        fc = np.zeros(6)
        kc[np.ix_(a, a)] = k[np.ix_(a, a)] - k[np.ix_(a, r)] @ np.linalg.solve(krr, k[np.ix_(r, a)])
        fc[a] = f[a] - k[np.ix_(a, r)] @ np.linalg.solve(krr, f[r])
        return kc, fc

    # ------------------------------------------------------------------ Auswertung
    def postprocess(self, d_global):
        """Lokale Verschiebungen (Gelenkverdrehungen zurückgerechnet) und Stabendkräfte aus den globalen Knotenwerten."""
        d = self.transformation() @ d_global
        k, f = self.local_stiffness(), self.local_equivalent_loads()
        r = self.released
        if r:
            a = [i for i in range(6) if i not in r]
            d = d.copy()
            d[r] = np.linalg.solve(k[np.ix_(r, r)], f[r] - k[np.ix_(r, a)] @ d[a])
        self.d_local = d
        self.end_forces = k @ d - f          # Kräfte der Knoten AUF den Stab (lokal): Fx1, Fy1, M1, Fx2, Fy2, M2
        return self

    def section(self, x):
        """
        Schnittgrössen und Verformungen an der lokalen Stelle x (Skalar oder Array, 0 ... L), exakt.

        Returns: dict mit N, Q, M, u, v, theta (lokal). u, v, theta sind die Verschiebungen in x_l, y_l und die Verdrehung.
        """
        x = np.asarray(x, dtype=float)
        L = self.length
        Fx1, Fy1, m1 = self.end_forces[:3]
        u1, v1, t1 = self.d_local[:3]
        qxa, qxb = self.qx
        qya, qyb = self.qy
        dqx, dqy = (qxb - qxa) / L, (qyb - qya) / L
        EA, EI = self.E * self.A, self.E * self.I
        N = -Fx1 - (qxa * x + dqx * x ** 2 / 2)
        Q = Fy1 + qya * x + dqy * x ** 2 / 2
        M = -m1 + Fy1 * x + qya * x ** 2 / 2 + dqy * x ** 3 / 6
        theta = t1 + (-m1 * x + Fy1 * x ** 2 / 2 + qya * x ** 3 / 6 + dqy * x ** 4 / 24) / EI
        v = v1 + t1 * x + (-m1 * x ** 2 / 2 + Fy1 * x ** 3 / 6 + qya * x ** 4 / 24 + dqy * x ** 5 / 120) / EI
        u = u1 + (-Fx1 * x - qxa * x ** 2 / 2 - dqx * x ** 3 / 6) / EA
        return dict(N=N, Q=Q, M=M, u=u, v=v, theta=theta)

    def extremes(self, points: int = 201):
        """Extremwerte von N, Q, M, |v| im Stab (M mit exakten Nullstellen der Querkraft)."""
        L = self.length
        qya, qyb = self.qy
        coef = np.trim_zeros([(qyb - qya) / (2 * L), qya, self.end_forces[1]], "f")
        cand = list(np.linspace(0, L, points))
        if len(coef) >= 2:
            cand += [float(r.real) for r in np.roots(coef) if abs(r.imag) < 1e-9 and 0 < r.real < L]
        x = np.array(sorted(cand))
        sec = self.section(x)
        out = {}
        for key in ("N", "Q", "M"):
            out[key + "_max"] = (float(sec[key].max()), float(x[sec[key].argmax()]))
            out[key + "_min"] = (float(sec[key].min()), float(x[sec[key].argmin()]))
        out["v_abs_max"] = (float(np.abs(sec["v"]).max()), float(x[np.abs(sec["v"]).argmax()]))
        return out


class FrameSystem(object):
    """
    Tragwerk aus Knoten und Stäben mit der Berechnung nach der direkten Steifigkeitsmethode.

    Ablauf: Knoten und Stäbe anlegen, FrameSystem(stäbe).solve(), danach Ergebnisse abfragen
    (displacements, reactions, element_forces, results, equilibrium_check, plot_data, plot, markdown).

    Attributes:
        elements (list): die Stäbe
        nodes (list): die Knoten in der Reihenfolge ihres ersten Auftretens (oder der übergebenen Liste)

    Nach solve():
        F (ndarray): globaler Lastvektor (Knotenlasten + äquivalente Knotenlasten der Streckenlasten)
        D (ndarray): globale Knotenverschiebungen [u_x, u_y, phi_z je Knoten]
        R (ndarray): globale Lagerreaktionen (gleiche Anordnung wie D)
        residual (float): Norm von K D - F - R (soll ~0 sein)
        auto_fixed (list): Knoten, deren unbestimmte Verdrehung (nur Gelenke) auf 0 gesetzt wurde
    """

    def __init__(self, elements: iter, nodes: iter = None) -> None:
        """
        Args:
            elements (Element | iterable): ein Stab oder eine Liste von Stäben
            nodes (iterable, optional): Knotenliste; sonst werden die Knoten aus den Stäben gesammelt

        Raises:
            ValueError: wenn ein Stab einen Knoten verwendet, der in einer übergebenen Knotenliste fehlt
        """
        self.elements = [elements] if isinstance(elements, Element) else list(elements)
        if nodes is None:
            self.nodes = []
            for el in self.elements:
                for n in (el.start, el.end):
                    if not any(n is m for m in self.nodes):
                        self.nodes.append(n)
        else:
            self.nodes = [nodes] if isinstance(nodes, Node) else list(nodes)
            for el in self.elements:
                for n in (el.start, el.end):
                    if not any(n is m for m in self.nodes):
                        raise ValueError(f"Stab {el.subscript}: Knoten {n.subscript} fehlt in der Knotenliste")
        for i, n in enumerate(self.nodes):
            n.index = i

    # ------------------------------------------------------------------ Lösung
    def solve(self):
        """
        Stellt K·D = F auf, berücksichtigt die Lager mit Lagrange-Multiplikatoren und löst.

        Danach verfügbar: D (globale Knotenverschiebungen), reactions(), element_forces(), results(), markdown() ...
        """
        nn = len(self.nodes)
        ndof = 3 * nn
        K = np.zeros((ndof, ndof))
        F = np.zeros(ndof)
        for n in self.nodes:
            sl = slice(3 * n.index, 3 * n.index + 3)
            F[sl] += n.force
            K[sl, sl] += np.diag(n.spring)
        for el in self.elements:
            kc, fc = el._condensed()
            T = el.transformation()
            dofs = np.r_[3 * el.start.index:3 * el.start.index + 3, 3 * el.end.index:3 * el.end.index + 3]
            K[np.ix_(dofs, dofs)] += T.T @ kc @ T
            F[dofs] += T.T @ fc
        self.F = F

        rows, gvals, owner = [], [], []
        for n in self.nodes:
            for (ax, ay, ar, val) in n.constraints:
                row = np.zeros(ndof)
                row[3 * n.index:3 * n.index + 3] = (ax, ay, ar)
                rows.append(row)
                gvals.append(val)
                owner.append((n, False))

        scale = float(np.max(np.abs(np.diag(K)))) or 1.0
        self.auto_fixed = []
        for n in self.nodes:
            i = 3 * n.index
            has_con = any(r[i:i + 3].any() for r in rows)
            for k in (0, 1):
                if abs(K[i + k, i + k]) < 1e-12 * scale and not has_con:
                    raise ValueError(f"Knoten {n.subscript} ist mit keinem Stab verbunden oder hat keine Steifigkeit in "
                                     f"{'xy'[k]}-Richtung.")
            if abs(K[i + 2, i + 2]) < 1e-12 * scale and not any(r[i + 2] != 0 for r in rows):
                if F[i + 2] != 0:
                    raise ValueError(f"Knoten {n.subscript}: Moment an einem Knoten, an dem alle Stäbe gelenkig anschliessen.")
                row = np.zeros(ndof)
                row[i + 2] = 1.0
                rows.append(row)
                gvals.append(0.0)
                owner.append((n, True))
                self.auto_fixed.append(n.subscript)

        m = len(rows)
        C = np.array(rows) if m else np.zeros((0, ndof))
        kap = scale                                      # Skalierung der Multiplikatoren, damit die Matrix gut konditioniert ist
        A = np.block([[K, kap * C.T], [kap * C, np.zeros((m, m))]])
        b = np.r_[F, kap * np.array(gvals)]
        with np.errstate(all="ignore"):
            cond = np.linalg.cond(A)
        if not np.isfinite(cond) or cond > 1e14:       # singuläre Systeme liefern >= 1e16
            raise ValueError("Tragwerk ist verschieblich (Mechanismus), zu wenig oder überbestimmt gelagert "
                             f"(Systemmatrix singulär, cond = {cond:.2g}).")
        sol = np.linalg.solve(A, b)
        self.D = sol[:ndof]
        mu = sol[ndof:]
        user = [k for k, (_, auto) in enumerate(owner) if not auto]      # automatisch gesperrte Verdrehungen sind keine Lager
        R = -kap * C[user].T @ mu[user] if user else np.zeros(ndof)      # Reaktionen der Lager
        for n in self.nodes:
            sl = slice(3 * n.index, 3 * n.index + 3)
            R[sl] -= n.spring * self.D[sl]                # Federreaktionen
        self.R = R
        self.residual = float(np.linalg.norm(K @ self.D - F - R))
        for el in self.elements:
            dofs = np.r_[3 * el.start.index:3 * el.start.index + 3, 3 * el.end.index:3 * el.end.index + 3]
            el.postprocess(self.D[dofs])
        return self

    # ------------------------------------------------------------------ Ergebnisse
    def displacements(self):
        """Array (Knoten, 3): ux, uy, rz je Knoten in der Reihenfolge von self.nodes."""
        return self.D.reshape(-1, 3)

    def reactions(self):
        """Liste der Lagerreaktionen: dict(node, x, y, Rx, Ry, Mz) für Knoten mit Lager oder Feder."""
        out = []
        for n in self.nodes:
            if n.constraints or n.spring.any():
                r = self.R[3 * n.index:3 * n.index + 3]
                out.append(dict(node=n.subscript, x=n.x, y=n.y, Rx=float(r[0]), Ry=float(r[1]), Mz=float(r[2])))
        return out

    def element_forces(self):
        """Array (Stäbe, 6): N(0), Q(0), M(0), N(L), Q(L), M(L) in der Schnittgrössenkonvention des Moduls."""
        out = []
        for el in self.elements:
            s0, s1 = el.section(0.0), el.section(el.length)
            out.append([float(s0["N"]), float(s0["Q"]), float(s0["M"]), float(s1["N"]), float(s1["Q"]), float(s1["M"])])
        return np.array(out)

    def equilibrium_check(self):
        """Summe aller Lasten und Reaktionen: Fx, Fy und Moment um den Ursprung (sollen 0 sein)."""
        tot = self.F + self.R
        fx, fy, mz = tot[0::3], tot[1::3], tot[2::3]
        xs = np.array([n.x for n in self.nodes])
        ys = np.array([n.y for n in self.nodes])
        return dict(sum_Fx=float(fx.sum()), sum_Fy=float(fy.sum()), sum_M=float((xs * fy - ys * fx + mz).sum()))

    def results(self):
        """Zusammenfassung: Verschiebungen, Reaktionen, Schnittgrössen-Extrema je Stab und global, Gleichgewicht."""
        disp = self.displacements()
        ext = {el.subscript: el.extremes() for el in self.elements}
        glob = {}
        for key in ("N_max", "N_min", "Q_max", "Q_min", "M_max", "M_min", "v_abs_max"):
            best = None
            for sub, e in ext.items():
                val = e[key][0]
                better = (best is None) or (val > best[0] if key.endswith("max") else val < best[0])
                if better:
                    best = (val, sub, e[key][1])
            glob[key] = dict(value=best[0], element=best[1], x=best[2])
        u_abs = np.hypot(disp[:, 0], disp[:, 1])
        k = int(u_abs.argmax())
        return dict(
            displacements=disp, reactions=self.reactions(), element_forces=self.element_forces(),
            extremes_per_element=ext, extremes=glob, equilibrium=self.equilibrium_check(),
            u_max=dict(value=float(u_abs[k]), node=self.nodes[k].subscript), auto_fixed=self.auto_fixed,
        )

    # ------------------------------------------------------------------ Diagrammdaten
    def plot_data(self, points: int = 51, scale: float = 1.0):
        """
        Verläufe für eigene Diagramme. Je Stab ein dict mit
            x (lokal), N, Q, M, u, v, theta (lokal), X0/Y0 (unverformt), X/Y (verformt, Verschiebungen mit scale überhöht)
        """
        out = []
        for el in self.elements:
            x = np.linspace(0, el.length, points)
            sec = el.section(x)
            ux = el.c * sec["u"] - el.s * sec["v"]
            uy = el.s * sec["u"] + el.c * sec["v"]
            X0, Y0 = el.start.x + el.c * x, el.start.y + el.s * x
            out.append(dict(element=el.subscript, x=x, N=sec["N"], Q=sec["Q"], M=sec["M"], u=sec["u"], v=sec["v"],
                            theta=sec["theta"], X0=X0, Y0=Y0, X=X0 + scale * ux, Y=Y0 + scale * uy))
        return out

    def diagram_curves(self, kind: str = "M", scale: float | None = None, points: int = 51):
        """
        Schnittgrössenlinie quer zum Stab aufgetragen (M auf der Zugseite). Liefert je Stab (X0, Y0, Xd, Yd) und den Faktor.
        scale: Länge je Einheit der Grösse [mm]; Standard: grösster Ausschlag = 10 % der Tragwerksgrösse.
        """
        kind = kind.upper()
        data = self.plot_data(points)
        vals = [d[kind] for d in data]
        vmax = max(float(np.abs(v).max()) for v in vals) or 1.0
        xs = [n.x for n in self.nodes]
        ys = [n.y for n in self.nodes]
        size = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        f = (0.1 * size / vmax) if scale is None else scale
        sign = -1.0 if kind == "M" else 1.0
        curves = []
        for el, d in zip(self.elements, data):
            off = sign * f * d[kind]
            curves.append((d["X0"], d["Y0"], d["X0"] - el.s * off, d["Y0"] + el.c * off))
        return curves, f

    def plot(self, kind: str = "M", scale: float | None = None, ax=None):
        """
        Zeichnet Tragwerk und Verlauf (optional, benötigt matplotlib): kind = "M", "N", "Q" oder "deformed".
        Returns: (fig, ax)
        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure
        for el in self.elements:
            ax.plot([el.start.x, el.end.x], [el.start.y, el.end.y], color="0.35", lw=2)
        for n in self.nodes:
            if n.constraints or n.spring.any():
                ax.plot(n.x, n.y, marker="^", color="tab:blue", ms=9)
        if kind.lower() == "deformed":
            xs = [n.x for n in self.nodes]
            ys = [n.y for n in self.nodes]
            size = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
            dmax = float(np.abs(self.displacements()[:, :2]).max()) or 1.0
            f = (0.1 * size / dmax) if scale is None else scale
            for d in self.plot_data(scale=f):
                ax.plot(d["X"], d["Y"], color="tab:red", lw=1.8)
            ax.set_title(f"Verformte Lage (Überhöhung x{f:.4g})")
        else:
            curves, f = self.diagram_curves(kind, scale)
            for (X0, Y0, Xd, Yd) in curves:
                ax.fill(np.r_[X0, Xd[::-1]], np.r_[Y0, Yd[::-1]], alpha=0.3, color="tab:orange")
                ax.plot(Xd, Yd, color="tab:orange", lw=1.5)
            unit = "Nmm" if kind.upper() == "M" else "N"
            ax.set_title(f"{kind.upper()}-Linie (1 {unit} = {f:.3g} mm)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        return fig, ax

    # ------------------------------------------------------------------ Markdown
    def markdown(self) -> str:
        """Ergebnisbericht als Markdown (Tabellen und LaTeX)."""
        r = self.results()
        lines = [
            "### Ebenes Rahmentragwerk",
            "---",
            f"{len(self.nodes)} Knoten, {len(self.elements)} Stäbe, {3 * len(self.nodes)} Freiheitsgrade. "
            r"Lösung der Gleichgewichtsbedingung $ K \, D = F $ mit der direkten Steifigkeitsmethode.",
            "Vorzeichen: Kräfte in $+x/+y$, Momente und Drehwinkel gegen den Uhrzeigersinn positiv. "
            r"$ N > 0 $ Zug; $ M > 0 $ Zugfaser auf der lokalen $-y_l$-Seite; $ Q' = q_y, \; M' = Q $.",
        ]
        if r["auto_fixed"]:
            lines.append(f"Hinweis: Die Verdrehung der Knoten {r['auto_fixed']} ist unbestimmt (nur Gelenke) und wurde auf 0 gesetzt.")

        tbl = ["| Knoten | x [mm] | y [mm] | $u_x$ [mm] | $u_y$ [mm] | $\\varphi_z$ [rad] |", "|:-:|--:|--:|--:|--:|--:|"]
        for n, d in zip(self.nodes, r["displacements"]):
            tbl.append(f"| {n.subscript} | {_g(n.x)} | {_g(n.y)} | {_g(d[0], 4)} | {_g(d[1], 4)} | {_g(d[2], 4)} |")
        lines += ["#### Knotenverschiebungen", "\n".join(tbl)]

        tbl = ["| Knoten | $R_x$ [N] | $R_y$ [N] | $M_z$ [Nmm] |", "|:-:|--:|--:|--:|"]
        for rc in r["reactions"]:
            tbl.append(f"| {rc['node']} | {_g(rc['Rx'], 5)} | {_g(rc['Ry'], 5)} | {_g(rc['Mz'], 5)} |")
        lines += ["#### Lagerreaktionen", "\n".join(tbl)]

        tbl = ["| Stab | Länge [mm] | Winkel [°] | $N_a$ [N] | $Q_a$ [N] | $M_a$ [Nmm] | $N_e$ [N] | $Q_e$ [N] | $M_e$ [Nmm] |",
               "|:-:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for el, f in zip(self.elements, r["element_forces"]):
            tbl.append(f"| {el.subscript} | {_g(el.length, 5)} | {_g(el.angle, 4)} | " + " | ".join(_g(v, 5) for v in f) + " |")
        lines += ["#### Schnittgrössen an den Stabenden (a = Anfang, e = Ende)", "\n".join(tbl)]

        e = r["extremes"]
        lines += [
            "#### Extremwerte",
            rf"Grösstes Moment (algebraisch): $ M_{{max}} = {_g(e['M_max']['value'])} \, \text{{Nmm}} $ in Stab {e['M_max']['element']} bei $ x_l = {_g(e['M_max']['x'], 5)} \, \text{{mm}} $",
            rf"Kleinstes Moment (algebraisch): $ M_{{min}} = {_g(e['M_min']['value'])} \, \text{{Nmm}} $ in Stab {e['M_min']['element']} bei $ x_l = {_g(e['M_min']['x'], 5)} \, \text{{mm}} $",
            rf"Normalkraft: $ N_{{max}} = {_g(e['N_max']['value'])} \, \text{{N}} $ in Stab {e['N_max']['element']} (Zug positiv), "
            rf"$ N_{{min}} = {_g(e['N_min']['value'])} \, \text{{N}} $ in Stab {e['N_min']['element']}",
            rf"Querkraft: $ Q_{{max}} = {_g(e['Q_max']['value'])} \, \text{{N}} $, $ Q_{{min}} = {_g(e['Q_min']['value'])} \, \text{{N}} $",
            rf"Grösste Durchbiegung quer zum Stab: $ |v|_{{max}} = {_g(e['v_abs_max']['value'], 4)} \, \text{{mm}} $ in Stab {e['v_abs_max']['element']}; "
            rf"grösste Knotenverschiebung $ {_g(r['u_max']['value'], 4)} \, \text{{mm}} $ an Knoten {r['u_max']['node']}",
            "#### Gleichgewichtskontrolle",
            rf"$ \sum F_x = {_g(r['equilibrium']['sum_Fx'], 3)} \, \text{{N}}, \quad \sum F_y = {_g(r['equilibrium']['sum_Fy'], 3)} \, \text{{N}}, \quad "
            rf"\sum M_0 = {_g(r['equilibrium']['sum_M'], 3)} \, \text{{Nmm}} $ (Lasten + Reaktionen, sollen 0 sein)",
        ]
        return "\n\n".join(lines)
