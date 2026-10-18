#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 23:13:23 2026

@author: fabian
"""

import numpy as np
from materials import Solid
from misc import _e, _f, _ok

_wiretypes_ = [["SL", "SM", "DM", "SH", "DH"], ["FD", "TD", "VD"]]


class Spring(object):
    def __init__(self, material: Solid, springRate: float):
        self.material = material
        self.springRate = springRate


class WireSpring(Spring):
    def __init__(
            self, material: Solid,
            Fmax: float,
            springRate: float,
            outerDiameter: float = 0,
            innerDiameter: float = 0,
            wiretype: str = "SL",
            formingtype: str = "cold"
    ):
        """
        Zylindrische Schraubendruckfeder mit Kreisquerschnitt.

        Args:
            material (Solid): Federwerkstoff (benötigt .shear = G in N/mm²)
            Fmax (float): grösste Federkraft F_n = F_2 [N]
            springRate (float): vorgegebene Federrate R_soll [N/mm]
            outerDiameter (float, optional): Aussendurchmesser D_e [mm], falls begrenzt.
            innerDiameter (float, optional): Innendurchmesser D_i [mm], falls begrenzt.
            wiretype (str, optional): Drahtsorte nach TB 10-2c. Defaults to "SL".
            formingtype (str, optional): "cold" oder "warm". Defaults to "cold".
        """
        super().__init__(material, springRate)
        if not (outerDiameter > 0 or innerDiameter > 0):
            raise ValueError("Entweder Aussen- oder Innendurchmesser angeben.")
        self.innerDiameter = innerDiameter
        self.outerDiameter = outerDiameter
        self.wiretype = wiretype
        self.Fmax = Fmax
        self.formingtype = formingtype
        # welcher Durchmesser vorgegeben ist, bleibt fest (calc_wireDiameter überschreibt den anderen)
        self._basis = "outer" if outerDiameter > 0 else "inner"

    # ------------------------------------------------------------------ Gl. 10.43
    def _k1(self, d: float) -> float:
        """Beiwert k1 nach Gl. 10.43 (abhängig von Drahtsorte und Drahtdurchmesser)."""
        if self.wiretype.upper() in _wiretypes_[0]:
            return 0.15 if d < 5 else 0.16
        return 0.17 if d < 5 else 0.18

    def calc_wireDiameter(self):
        """
        Drahtdurchmesser überschlägig nach Gl. 10.43.
        k1 hängt vom (noch unbekannten) d ab: erst mit d < 5 mm rechnen, bei d >= 5 mm mit dem zweiten Wert nochmals.

        Returns:
            float: Drahtdurchmesser d' [mm] (anschliessend nach TB 10-2a wählen)
        """
        F = self.Fmax

        def d_of(k1):
            if self._basis == "outer":
                return k1 * (F * self.outerDiameter) ** (1 / 3), 0.0
            base = k1 * (F * self.innerDiameter) ** (1 / 3)
            k2 = 2 * base ** 2 / (3 * self.innerDiameter)
            return base + k2, k2

        k1 = self._k1(0)
        d, k2 = d_of(k1)
        if d >= 5:
            k1 = self._k1(5)
            d, k2 = d_of(k1)
        self.k1, self.k2, self.wireDiameter = k1, k2, d

        if self._basis == "outer":
            self.innerDiameter = self.outerDiameter - 2 * d
        else:
            self.outerDiameter = self.innerDiameter + 2 * d
        assert self.innerDiameter > 0, "Aussendurchmesser zu klein: Feder kann Fmax nicht aufnehmen."
        self.nominalDiameter = (self.outerDiameter + self.innerDiameter) / 2  # D = D_e - d = D_i + d
        return self.wireDiameter

    # ------------------------------------------------------------------ Gln. 10.37 ... 10.49
    def calc_spring(
            self, truewireDiameter: float, maxWireDiameter: float,
            k: float | None = None, F1: float = 0.0, D: float | None = None,
            ends: str = "ground", windings: float | None = None,
            Rm: float | None = None, tau_kO: float | None = None, tau_kH: float | None = None
    ):
        """
        Federberechnung nach Roloff/Matek.

        Args:
            truewireDiameter (float): gewählter Drahtdurchmesser d [mm] (TB 10-2a)
            maxWireDiameter (float): d_max = d + e_s [mm] (TB 10-2a)
            k (float, optional): Spannungsbeiwert (TB 10-15b). Default: k = (w + 0.5)/(w - 0.75).
            F1 (float, optional): Unterkraft [N]; 0 = von Null schwellend (konservativ).
            D (float, optional): gewählter mittlerer Windungsdurchmesser [mm] (DIN 323 R20', TB 1-16).
                Default: D = D_e - d bzw. D_i + d aus dem vorgegebenen Durchmesser.
            ends (str): "ground" (angelegt, plangeschliffen) oder "unground" (angelegt, unbearbeitet), Gl. 10.39.
            windings (float, optional): gewählte federnde Windungszahl n. Default: auf ganze Zahl gerundet.
                Bei dynamischer Beanspruchung n_t auf ,5 enden lassen (z.B. n = 5.5 kalt).
            Rm (float, optional): Zugfestigkeit des Drahts [N/mm²] (TB 10-3) für die statischen Nachweise.
            tau_kO, tau_kH (float, optional): Dauerfestigkeitswerte aus TB 10-16 ... 10-19 [N/mm²].
        """
        d = self.truewireDiameter = truewireDiameter
        self.maxWireDiameter = maxWireDiameter
        warm = self.formingtype.lower() == "warm"
        ground = ends.lower() == "ground"
        self.ends = "ground" if ground else "unground"
        G = self.material.shear

        # D = (D_e + D_i)/2 = D_e - d = D_i + d
        if D is None:
            D = self.outerDiameter - d if self._basis == "outer" else self.innerDiameter + d
        self.nominalDiameter = D
        self.outerDiameter, self.innerDiameter = D + d, D - d
        self.w = D / d

        # Windungszahl (10.46), Federrate (10.47), Federweg (10.49)
        self.windingsRequired = G * d ** 4 / (8 * D ** 3 * self.springRate)
        self.windings = round(self.windingsRequired) if windings is None else windings
        n = self.windings
        assert n >= (3 if warm else 2), f"n = {n:g} zu klein (kalt: n >= 2, warm: n >= 3)."
        self.trueSpringRate = G * d ** 4 / (8 * D ** 3 * n)
        self.trueSpringPitch = 8 * D ** 3 * n * self.Fmax / (G * d ** 4)  # Federweg s_n

        # Gesamtwindungszahl (10.37), Mindestabstände (10.38), Blocklänge (10.39), Länge Ln (10.40)
        if warm:
            self.trueWindings = n + 1.5
            self.blockCoef = -0.3 if ground else 1.1
            self.Sa = 0.02 * (D + d) * n
            self.Sa_dyn = 2 * self.Sa
        else:
            self.trueWindings = n + 2
            self.blockCoef = 0.0 if ground else 1.5
            self.Sa = (0.0015 * D ** 2 / d + 0.1 * d) * n
            self.Sa_dyn = 1.5 * self.Sa
        self.blockPitch = (self.trueWindings + self.blockCoef) * maxWireDiameter  # L_c
        self.maxPitch = self.blockPitch + self.Sa  # L_n
        self.maxPitch_dyn = self.blockPitch + self.Sa_dyn  # L_n'

        # unbelastete Länge und Blockzustand (10.41)
        self.sc = self.trueSpringPitch + self.Sa
        self.sc_dyn = self.trueSpringPitch + self.Sa_dyn
        self.L0 = self.sc + self.blockPitch
        self.L0_dyn = self.sc_dyn + self.blockPitch
        self.Fc = self.trueSpringRate * self.sc
        self.Fc_dyn = self.trueSpringRate * self.sc_dyn

        # Schubspannungen (10.44), korrigiert (10.45)
        W = np.pi * d ** 3 / 16
        self.F1 = F1
        self.k = (self.w + 0.5) / (self.w - 0.75) if k is None else k
        self.tau_12 = np.array([F1, self.Fmax]) * D / (2 * W)
        self.tau_k12 = self.k * self.tau_12
        self.tau_kh = abs(np.diff(self.tau_k12))
        self.tau_c = self.Fc * D / (2 * W)
        self.tau_c_dyn = self.Fc_dyn * D / (2 * W)

        # zulässige Werte (TB 10-1, TB 10-15a)
        self.Rm, self.tau_kO, self.tau_kH = Rm, tau_kO, tau_kH
        self.tau_zul = 0.5 * Rm if Rm is not None else None
        if warm:
            self.tau_c_zul = 1180 - 262 * np.log10(d)
        else:
            self.tau_c_zul = 0.56 * Rm if Rm is not None else None

    # ------------------------------------------------------------------ Markdown
    def markdown_wireDiameter(self) -> str:
        """Drahtdurchmesser-Vorberechnung nach Gl. 10.43 (nach calc_wireDiameter())."""
        F = self.Fmax
        N, mm = r"\, \text{N}", r"\, \text{mm}"
        sign = "<" if self.wireDiameter < 5 else r"\geq"
        lines = [
            "### Drahtdurchmesser (Vorberechnung, Gl. 10.43)",
            "---",
            rf"Drahtsorte **{self.wiretype.upper()}**: $ k_1 = {self.k1} $ (für $ d {sign} 5 {mm} $)",
            rf"Grösste Federkraft: $ F_{{max}} = {_f(F)} {N} $",
        ]
        if self._basis == "outer":
            Da = self.outerDiameter
            lines += [
                rf"Gegeben ist der Aussendurchmesser: $ D_e = {Da:g} {mm} $",
                rf"Drahtdurchmesser: $ d \approx k_1 \sqrt[3]{{F \, D_e}} = {self.k1} \cdot \sqrt[3]{{{_f(F)} \cdot {Da:g}}} = {self.wireDiameter:.3f} {mm} $",
                rf"Innendurchmesser: $ D_i = D_e - 2d = {Da:g} - 2 \cdot {self.wireDiameter:.3f} = {self.innerDiameter:.3f} {mm} $",
            ]
        else:
            Di = self.innerDiameter
            base = self.wireDiameter - self.k2
            lines += [
                rf"Gegeben ist der Innendurchmesser: $ D_i = {Di:g} {mm} $",
                rf"Hilfsgrösse: $ k_2 \approx \frac{{2 \left( k_1 \sqrt[3]{{F \, D_i}} \right)^2}}{{3 D_i}} = \frac{{2 \cdot {base:.3f}^2}}{{3 \cdot {Di:g}}} = {self.k2:.3f} {mm} $",
                rf"Drahtdurchmesser: $ d \approx k_1 \sqrt[3]{{F \, D_i}} + k_2 = {base:.3f} + {self.k2:.3f} = {self.wireDiameter:.3f} {mm} $",
                rf"Aussendurchmesser: $ D_e = D_i + 2d = {self.outerDiameter:.3f} {mm} $",
            ]
        lines += [
            rf"Mittlerer Windungsdurchmesser: $ D = \frac{{D_e + D_i}}{{2}} = {self.nominalDiameter:.3f} {mm} $",
            "Nun $ d $ nach TB 10-2a wählen und $ D $ nach DIN 323 (R20', TB 1-16) festlegen.",
        ]
        return "\n\n".join(lines)

    def markdown(self) -> str:
        """Vollständige Federberechnung (nach calc_spring())."""
        d, D, G = self.truewireDiameter, self.nominalDiameter, self.material.shear
        n, nt, F2 = self.windings, self.trueWindings, self.Fmax
        warm = self.formingtype.lower() == "warm"
        N, mm = r"\, \text{N}", r"\, \text{mm}"
        MPa = r" \, \frac{\text{N}}{\text{mm}^2}"
        Nmm = r" \, \frac{\text{N}}{\text{mm}}"
        dmax = self.maxWireDiameter
        w_lo, w_hi = (3, 12) if warm else (4, 20)

        if self.blockCoef == 0:
            Lc_form = r"n_t \, d_{max}"
        else:
            sg = "+" if self.blockCoef > 0 else "-"
            Lc_form = rf"(n_t {sg} {abs(self.blockCoef)}) \, d_{{max}}"
        if warm:
            nt_txt = rf"n_t = n + 1.5 = {n:g} + 1.5 = {nt:g}"
            Sa_form = rf"S_a = 0.02 \, (D + d) \, n = 0.02 \cdot ({D:.2f} + {d:g}) \cdot {n:g}"
            Sad_form = r"S_a' \approx 2 \, S_a"
        else:
            nt_txt = rf"n_t = n + 2 = {n:g} + 2 = {nt:g}"
            Sa_form = (rf"S_a = \left( 0.0015 \frac{{D^2}}{{d}} + 0.1 \, d \right) n = "
                       rf"\left( 0.0015 \cdot \frac{{{D:.2f}^2}}{{{d:g}}} + 0.1 \cdot {d:g} \right) \cdot {n:g}")
            Sad_form = r"S_a' \approx 1.5 \, S_a"
        tau_kh = float(np.ravel(self.tau_kh)[0])
        ends_txt = "angelegt und plangeschliffen" if self.ends == "ground" else "angelegt und unbearbeitet"

        lines = [
            f"### Zylindrische Schraubendruckfeder ({'warm' if warm else 'kalt'}geformt, Enden {ends_txt})",
            "---",
            "#### Geometrie",
            rf"Drahtdurchmesser: $ d = {d:g} {mm}, \quad d_{{max}} = d + e_s = {dmax:g} {mm} $",
            rf"Mittlerer Windungsdurchmesser: $ D = D_e - d = D_i + d = {D:.2f} {mm} $ "
            rf"(Aussen $ D_e = {self.outerDiameter:.2f} {mm} $, innen $ D_i = {self.innerDiameter:.2f} {mm} $)",
            rf"Wickelverhältnis: $ w = \frac{{D}}{{d}} = {self.w:.2f} $ "
            rf"(zulässig {w_lo} ... {w_hi}) {_ok(w_lo <= self.w <= w_hi)}",
            rf"Schubmodul: $ G = {_f(G)} {MPa} $",

            "#### Windungszahl und Federrate",
            rf"Erforderliche Windungszahl: $ n' = \frac{{G}}{{8}} \frac{{d^4}}{{D^3 \, R_{{soll}}}} = \frac{{{_f(G)}}}{{8}} \cdot \frac{{{d:g}^4}}{{{D:.2f}^3 \cdot {self.springRate:g}}} = {self.windingsRequired:.2f} $",
            rf"Gewählt: $ n = {n:g} $",
            rf"Federrate: $ R_{{ist}} = \frac{{G}}{{8}} \frac{{d^4}}{{D^3 \, n}} = {self.trueSpringRate:.3f} {Nmm} $",
            rf"Federweg bei $ F_n $: $ s_n = \frac{{F_n}}{{R_{{ist}}}} = \frac{{8 \, D^3 \, n \, F_n}}{{G \, d^4}} = {self.trueSpringPitch:.2f} {mm} $",

            "#### Federlängen",
            rf"Gesamtwindungszahl (Gl. 10.37): $ {nt_txt} $",
            rf"Blocklänge (Gl. 10.39): $ L_c = {Lc_form} = {self.blockPitch:.2f} {mm} $",
            rf"Mindestabstände (Gl. 10.38a): $ {Sa_form} = {self.Sa:.2f} {mm} $",
            rf"Kleinste Federlänge (Gl. 10.40): $ L_n = L_c + S_a = {self.blockPitch:.2f} + {self.Sa:.2f} = {self.maxPitch:.2f} {mm} $",
            rf"Bei dynamischer Beanspruchung (Gl. 10.38b): $ {Sad_form} = {self.Sa_dyn:.2f} {mm}, \quad L_n' = {self.maxPitch_dyn:.2f} {mm} $",
            rf"Unbelastete Länge (Gl. 10.41): $ L_0 = s_n + L_c + S_a = {self.L0:.2f} {mm} $ (dynamisch: $ {self.L0_dyn:.2f} {mm} $)",
            rf"Federweg im Blockzustand: $ s_c = s_n + S_a = {self.sc:.2f} {mm}, \quad F_c = R_{{ist}} \, s_c = {_f(self.Fc)} {N} $",

            "#### Schubspannungen (Gl. 10.44)",
            rf"Kräfte: $ F_1 = {_f(self.F1, 1)} {N}, \quad F_2 = F_n = {_f(F2)} {N} $",
            rf"$ \tau_{{1,2}} = \frac{{F_{{1,2}} \cdot D/2}}{{\frac{{\pi}}{{16}} d^3}} = [{self.tau_12[0]:.1f}, {self.tau_12[1]:.1f}] {MPa} $",
            rf"Blockzustand: $ \tau_c = \frac{{F_c \cdot D/2}}{{\frac{{\pi}}{{16}} d^3}} = {self.tau_c:.1f} {MPa} $",
        ]
        if self.tau_zul is not None:
            lines.append(
                rf"Statisch: $ \tau_2 = {self.tau_12[1]:.1f} \leq \tau_{{zul}} = 0.5 \, R_m = {self.tau_zul:.1f} {MPa} $ → {_ok(self.tau_12[1] <= self.tau_zul)}")
        if self.tau_c_zul is not None:
            zul_txt = (r"1180 - 262 \lg d" if warm else r"0.56 \, R_m")
            lines.append(
                rf"Blockzustand: $ \tau_c = {self.tau_c:.1f} \leq \tau_{{c,zul}} = {zul_txt} = {self.tau_c_zul:.1f} {MPa} $ → {_ok(self.tau_c <= self.tau_c_zul)}")

        if self.tau_kO is not None or self.tau_kH is not None:
            lines += [
                "#### Dynamische Beanspruchung (Gl. 10.45)",
                rf"Spannungsbeiwert (TB 10-15b): $ k = \frac{{w + 0.5}}{{w - 0.75}} = {self.k:.3f} $",
                rf"Korrigierte Schubspannungen: $ \tau_{{k1,2}} = k \, \tau_{{1,2}} = [{self.tau_k12[0]:.1f}, {self.tau_k12[1]:.1f}] {MPa} $",
                rf"Korrigierte Hubspannung: $ \tau_{{kh}} = \tau_{{k2}} - \tau_{{k1}} = {tau_kh:.1f} {MPa} $",
            ]
            if self.tau_kO is not None:
                lines.append(
                    rf"$ \tau_{{k2}} = {self.tau_k12[1]:.1f} \leq \tau_{{kO}} = {self.tau_kO:.1f} {MPa} $ → {_ok(self.tau_k12[1] <= self.tau_kO)}")
            if self.tau_kH is not None:
                lines.append(
                    rf"$ \tau_{{kh}} = {tau_kh:.1f} \leq \tau_{{kH}} = {self.tau_kH:.1f} {MPa} $ → {_ok(tau_kh <= self.tau_kH)}")
            lines.append(
                rf"Blockzustand bei dynamischer Auslegung: $ \tau_c' = {self.tau_c_dyn:.1f} {MPa} $ mit $ L_0' = {self.L0_dyn:.2f} {mm} $")
        return "\n\n".join(lines)


# TB 10-9: (De, Di, t, t' oder None, h0, F_0,75) in mm bzw. N.   l0 = t + h0
_TB10_9_ = {
    "A": [  # De/t ~ 18, h0/t ~ 0,4
        (8, 4.2, .4, None, .2, 210), (10, 5.2, .5, None, .25, 325), (12.5, 6.2, .7, None, .3, 660),
        (14, 7.2, .8, None, .3, 797), (16, 8.2, .9, None, .35, 1013), (18, 9.2, 1, None, .4, 1254),
        (20, 10.2, 1.1, None, .45, 1521),
        (22.5, 11.2, 1.25, None, .5, 1929), (25, 12.2, 1.5, None, .55, 2926), (28, 14.2, 1.5, None, .65, 2841),
        (31.5, 16.3, 1.75, None, .7, 3871), (35.5, 18.3, 2, None, .8, 5187), (40, 20.4, 2.25, None, .9, 6500),
        (45, 22.4, 2.5, None, 1, 7716), (50, 25.4, 3, None, 1.1, 11976), (56, 28.5, 3, None, 1.3, 11388),
        (63, 31, 3.5, None, 1.4, 15025), (71, 36, 4, None, 1.6, 20535), (80, 41, 5, None, 1.7, 33559),
        (90, 46, 5, None, 2, 31354), (100, 51, 6, None, 2.2, 48022), (112, 57, 6, None, 2.5, 43707),
        (125, 64, 8, 7.5, 2.6, 85926), (140, 72, 8, 7.5, 3.2, 85251), (160, 82, 10, 9.4, 3.5, 138331),
        (180, 92, 10, 9.4, 4, 125417), (200, 102, 12, 11.25, 4.2, 183020), (225, 112, 12, 11.25, 5, 171016),
        (250, 127, 14, 13.1, 5.6, 248828),
    ],
    "B": [  # De/t ~ 28, h0/t ~ 0,75
        (8, 4.2, .3, None, .25, 118), (10, 5.2, .4, None, .3, 209), (12.5, 6.2, .5, None, .35, 294),
        (14, 7.2, .5, None, .4, 279), (16, 8.2, .6, None, .45, 410), (18, 9.2, .7, None, .5, 566),
        (20, 10.2, .8, None, .55, 748), (22.5, 11.2, .8, None, .65, 707), (25, 12.2, .9, None, .7, 862),
        (28, 14.2, 1, None, .8, 1107),
        (31.5, 16.3, 1.25, None, .9, 1913), (35.5, 18.3, 1.25, None, 1, 1699), (40, 20.4, 1.5, None, 1.15, 2622),
        (45, 22.4, 1.75, None, 1.3, 3646), (50, 25.4, 2, None, 1.4, 4762), (56, 28.5, 2, None, 1.6, 4438),
        (63, 31, 2.5, None, 1.75, 7189), (71, 36, 2.5, None, 2, 6725), (80, 41, 3, None, 2.3, 10518),
        (90, 46, 3.5, None, 2.5, 14161), (100, 51, 3.5, None, 2.8, 13070), (112, 57, 4, None, 3.2, 17752),
        (125, 64, 5, None, 3.5, 29908), (140, 72, 5, None, 4, 27920), (160, 82, 6, None, 4.5, 41008),
        (180, 92, 6, None, 5.1, 37502),
        (200, 102, 8, 7.5, 5.6, 76378), (225, 112, 8, 7.5, 6.5, 70749), (250, 127, 10, 9.4, 7, 119050),
    ],
    "C": [  # De/t ~ 40, h0/t ~ 1,3
        (8, 4.2, .2, None, .25, 39), (10, 5.2, .25, None, .3, 58), (12.5, 6.2, .35, None, .45, 151),
        (14, 7.2, .35, None, .45, 123), (16, 8.2, .4, None, .5, 154), (18, 9.2, .45, None, .6, 214),
        (20, 10.2, .5, None, .65, 254), (22.5, 11.2, .6, None, .8, 426), (25, 12.2, .7, None, .9, 600),
        (28, 14.2, .8, None, 1, 801), (31.5, 16.3, .8, None, 1.05, 687), (35.5, 18.3, .9, None, 1.15, 832),
        (40, 20.4, 1, None, 1.3, 1017),
        (45, 22.4, 1.25, None, 1.6, 1891), (50, 25.4, 1.25, None, 1.6, 1550), (56, 28.5, 1.5, None, 1.95, 2622),
        (63, 31, 1.8, None, 2.35, 4238), (71, 36, 2, None, 2.6, 5144), (80, 41, 2.25, None, 2.95, 6613),
        (90, 46, 2.5, None, 3.2, 7684), (100, 51, 2.7, None, 3.5, 8609), (112, 57, 3, None, 3.9, 10489),
        (125, 64, 3.5, None, 4.5, 15416), (140, 72, 3.8, None, 4.9, 17195), (160, 82, 4.3, None, 5.6, 21843),
        (180, 92, 4.8, None, 6.2, 26442), (200, 102, 5.5, None, 7, 36111),
        (225, 112, 6.5, 6.2, 7.1, 44580), (250, 127, 7, 6.7, 7.8, 50466),
    ],
}


class DiscSpring(object):
    def __init__(self, De: float, Di: float, t: float, l0: float,
                 E: float = 206_000, mu: float = 0.3,
                 t_reduced: float | None = None, series: str | None = None):
        """
        Einzelteller nach DIN EN 16983.

        Tellerfedern nach Roloff/Matek Kap. 10.3.5 (Gln. 10.22 bis 10.32, TB 10-9, TB 10-11, TB 10-13).

        Geprüft gegen TB 10-9: Einzelteller der Gruppen 1 und 2 (t <= 6 mm) reproduzieren F_0,75, sigma_OM und die
        grösste Zugspannung (sigma_II bei Reihe A, sigma_III bei Reihe B und C) auf 0,25 % genau.
        Gruppe 3 (t > 6 mm, Auflageflächen, reduzierte Dicke t'): das Buch verlangt den Beiwert K4 (Gl. 10.25 ff.). Er
        ist hier NICHT umgesetzt (Formel im Quelltext nicht eindeutig lesbar). Gruppe 3 wird mit den Nennmassen t, h0
        gerechnet. Das reproduziert F_0,75 aus TB 10-9 genau, die Spannungen sigma_II/III weichen aber um 5 ... 11 %
        vom Tabellenwert ab (sigma_OM ca. 1,5 %). Solche Teller bitte mit Herstellerdaten gegenprüfen.

        Ablauf:
        >>> disc = DiscSpring.from_standard("A", 25)                    # DIN EN 16983 Reihe A, De = 25 mm
        >>> stack = DiscSpringStack(disc, n=1, i=18, wM=0.02, wR=0.03)
        >>> stack.calc_for_force(2500, s_required=6, friction=True)
        >>> display(Markdown(disc.markdown(s=stack.s_single)))
        >>> display(Markdown(stack.markdown()))


        Args:
            De (float): Aussendurchmesser [mm]
            Di (float): Innendurchmesser [mm]
            t (float): Tellerdicke [mm]
            l0 (float): Bauhöhe des unbelasteten Tellers [mm]
            E (float): Elastizitätsmodul [N/mm²] (Federstahl 206 000)
            mu (float): Querkontraktionszahl (Federstahl 0,3)
            t_reduced (float, optional): reduzierte Dicke t' (Gruppe 3), nur zur Anzeige
            series (str, optional): "A", "B" oder "C" (nur zur Anzeige)
        """
        assert De > Di > 0 and t > 0 and l0 > t, "Es muss De > Di > 0 und l0 > t gelten."
        self.De, self.Di, self.t, self.l0 = De, Di, t, l0
        self.E, self.mu, self.t_reduced, self.series = E, mu, t_reduced, series
        self.h0 = l0 - t
        self.delta = De / Di
        self.group = 1 if t < 1.25 else (2 if t <= 6 else 3)
        self.approximate = self.group == 3  # K4 nicht umgesetzt, s. Modulkopf
        self.F075_table = None

        d, ln = self.delta, np.log(self.delta)
        self.K1 = (1 / np.pi) * ((d - 1) / d) ** 2 / ((d + 1) / (d - 1) - 2 / ln)
        self.K2 = (6 / np.pi) * ((d - 1) / ln - 1) / ln
        self.K3 = (3 / np.pi) * (d - 1) / ln
        self._A = 4 * E / (1 - mu ** 2) * t ** 4 / (self.K1 * De ** 2)  # Vorfaktor der Gl. 10.25

    @classmethod
    def from_standard(cls, series: str, De: float, E: float = 206_000, mu: float = 0.3):
        """Teller aus TB 10-9 (Reihe A, B oder C) über den Aussendurchmesser."""
        for (De_, Di, t, tp, h0, F075) in _TB10_9_[series.upper()]:
            if abs(De_ - De) < 1e-9:
                obj = cls(De_, Di, t, t + h0, E, mu, tp, series.upper())
                obj.F075_table = F075
                return obj
        raise ValueError(f"Reihe {series}: De = {De} mm steht nicht in TB 10-9.")

    # ------------------------------------------------------------------ Kennlinie
    @property
    def s075(self):
        return 0.75 * self.h0

    def force(self, s: float) -> float:
        """Federkraft des Einzeltellers bei Federweg s, Gl. 10.25 [N]."""
        t, h0 = self.t, self.h0
        return self._A * (s / t) * ((h0 / t - s / t) * (h0 / t - s / (2 * t)) + 1)

    @property
    def Fc(self) -> float:
        """Federkraft in Planlage, Gl. 10.26 [N]."""
        return self.force(self.h0)

    @property
    def F075(self) -> float:
        return self.force(self.s075)

    def rate(self, s: float) -> float:
        """Federrate R = dF/ds bei Federweg s, Gl. 10.27 [N/mm]."""
        t, a, x = self.t, self.h0 / self.t, s / self.t
        return self._A / t * (a ** 2 - 3 * a * x + 1.5 * x ** 2 + 1)

    def work(self, s: float) -> float:
        """Federungsarbeit bei Einfederung s, Gl. 10.28 [Nmm]."""
        t, a, x = self.t, self.h0 / self.t, s / self.t
        return self._A * t / 2 * x ** 2 * ((a - x / 2) ** 2 + 1)

    def travel_for_force(self, F: float) -> float:
        """Federweg s zu gegebener Kraft F (Umkehrung von Gl. 10.25, Bereich 0 ... 0,75 h0)."""
        if F < 0 or F > self.F075 * (1 + 1e-9):
            raise ValueError(f"F = {F:.0f} N liegt ausserhalb 0 ... F_0,75 = {self.F075:.0f} N.")
        lo, hi = 0.0, self.s075
        for _ in range(100):
            mid = (lo + hi) / 2
            if self.force(mid) < F:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    # ------------------------------------------------------------------ Spannungen
    def stresses(self, s: float) -> dict:
        """Rechnerische Spannungen nach Gl. 10.29 und 10.30 [N/mm²] (Zug positiv)."""
        t, a, x, d = self.t, self.h0 / self.t, s / self.t, self.delta
        U = 4 * self.E / (1 - self.mu ** 2) * t ** 2 / (self.K1 * self.De ** 2) * x
        b = a - x / 2
        return {
            "OM": -U * 3 / np.pi,
            "I": -U * (self.K2 * b + self.K3),
            "II": -U * (self.K2 * b - self.K3),
            "III": -U / d * ((self.K2 - 2 * self.K3) * b - self.K3),
            "IV": -U / d * ((self.K2 - 2 * self.K3) * b + self.K3),
        }

    def tension_max(self, s: float) -> float:
        """Grösste Zugspannung (Stelle II oder III), von der Brüche ausgehen."""
        st = self.stresses(s)
        return max(st["II"], st["III"])

    def fatigue_check(self, s1: float, s2: float, sigma_H: float | None = None) -> dict:
        """
        Dynamischer Nachweis (Gl. 10.31): Hubspannung sigma_h = sigma_2 - sigma_1 <= sigma_H.
        sigma_H (Dauerhubfestigkeit) aus TB 10-12 ablesen. Zusätzlich |sigma_I(s1)| >= 600 N/mm² anstreben.
        """
        sig1, sig2 = self.tension_max(s1), self.tension_max(s2)
        res = {"sigma_1": sig1, "sigma_2": sig2, "sigma_h": sig2 - sig1,
               "sigma_I_1": self.stresses(s1)["I"], "sigma_H": sigma_H}
        res["preload_ok"] = res["sigma_I_1"] <= -600
        res["ok"] = None if sigma_H is None else res["sigma_h"] <= sigma_H
        return res

    # ------------------------------------------------------------------ Markdown
    def markdown(self, s: float | None = None) -> str:
        """Einzelteller: Geometrie, Kennwerte, Kraft, Federrate, Arbeit und Spannungen bei Federweg s."""
        s = self.s075 if s is None else s
        mm, N = r"\, \text{mm}", r"\, \text{N}"
        MPa = r" \, \frac{\text{N}}{\text{mm}^2}"
        t, h0, d = self.t, self.h0, self.delta
        sr = f" Reihe {self.series}," if self.series else ""
        st = self.stresses(s)
        lines = [
            f"### Tellerfeder DIN EN 16983,{sr} Gruppe {self.group}",
            "---",
            "#### Geometrie",
            rf"$ D_e = {self.De:g} {mm}, \quad D_i = {self.Di:g} {mm}, \quad t = {t:g} {mm}" +
            (rf", \quad t' = {self.t_reduced:g} {mm}" if self.t_reduced else "") + rf", \quad l_0 = {self.l0:g} {mm} $",
            rf"Federweg bis zur Planlage: $ h_0 = l_0 - t = {h0:g} {mm}, \quad \frac{{h_0}}{{t}} = {h0 / t:.2f}, \quad \frac{{D_e}}{{t}} = {self.De / t:.1f} $",
            rf"Durchmesserverhältnis: $ \delta = \frac{{D_e}}{{D_i}} = {d:.3f} $",
            rf"Kennwerte (TB 10-11): $ K_1 = {self.K1:.4f}, \quad K_2 = {self.K2:.4f}, \quad K_3 = {self.K3:.4f} $",

            "#### Kennlinie",
            rf"Federkraft bei $ s = {s:.3f} {mm} $ (Gl. 10.25): $ F = \frac{{4E}}{{1 - \mu^2}} \frac{{t^4}}{{K_1 D_e^2}} \frac{{s}}{{t}} \left[ \left( \frac{{h_0}}{{t}} - \frac{{s}}{{t}} \right) \left( \frac{{h_0}}{{t}} - \frac{{s}}{{2t}} \right) + 1 \right] = {_f(self.force(s), 1)} {N} $",
            rf"Kraft in Planlage (Gl. 10.26): $ F_c = \frac{{4E}}{{1 - \mu^2}} \frac{{h_0 \, t^3}}{{K_1 D_e^2}} = {_f(self.Fc)} {N} $",
            rf"Grenzwert $ s_{{0,75}} = 0.75 \, h_0 = {self.s075:.3f} {mm}, \quad F_{{0,75}} = {_f(self.F075)} {N} $"
            + (rf" (TB 10-9: ${_f(self.F075_table)} {N}$)" if self.F075_table else ""),
            rf"Federrate (Gl. 10.27): $ R = \frac{{dF}}{{ds}} = {self.rate(s):.1f} \, \frac{{\text{{N}}}}{{\text{{mm}}}} $",
            rf"Federungsarbeit (Gl. 10.28): $ W = {_f(self.work(s))} \, \text{{Nmm}} $",

            "#### Spannungen (Gln. 10.29, 10.30)",
            rf"$ \sigma_{{OM}} = {st['OM']:.0f} {MPa}, \quad \sigma_I = {st['I']:.0f} {MPa}, \quad \sigma_{{II}} = {st['II']:.0f} {MPa} $",
            rf"$ \sigma_{{III}} = {st['III']:.0f} {MPa}, \quad \sigma_{{IV}} = {st['IV']:.0f} {MPa} $",
            (rf"Statisch: $ s = {s:.3f} \leq s_{{0,75}} = {self.s075:.3f} {mm} $ → ✔ (kein weiterer Nachweis nötig)"
             if s <= self.s075 + 1e-12 else
             rf"Statisch: $ s > s_{{0,75}} $ → $ |\sigma_{{OM}}| = {abs(st['OM']):.0f} {MPa} $ gegen $ R_e = 1400 \ldots 1600 {MPa} $ prüfen"),
        ]
        if self.approximate:
            lines.append(
                "**Hinweis:** Gruppe 3 (t > 6 mm) ist ohne den Beiwert K4 gerechnet. Kraft stimmt mit TB 10-9 überein, "
                "die Zugspannungen $ \\sigma_{II}, \\sigma_{III} $ weichen um bis zu etwa 11 % ab. Mit Herstellerdaten prüfen.")
        return "\n\n".join(lines)

    def markdown_dynamic(self, s1: float, s2: float, sigma_H: float | None = None) -> str:
        """Dynamischer Nachweis (Gl. 10.31)."""
        r = self.fatigue_check(s1, s2, sigma_H)
        MPa = r" \, \frac{\text{N}}{\text{mm}^2}"
        lines = [
            "#### Dynamischer Nachweis (Gl. 10.31)",
            rf"Vorspannweg $ s_1 = {s1:.3f} \, \text{{mm}} $, Hubende $ s_2 = {s2:.3f} \, \text{{mm}} $",
            rf"Zugspannungen an der Tellerunterseite (Stelle II bzw. III): $ \sigma_1 = {r['sigma_1']:.0f} {MPa}, \quad \sigma_2 = {r['sigma_2']:.0f} {MPa} $",
            rf"Hubspannung: $ \sigma_h = \sigma_2 - \sigma_1 = {r['sigma_h']:.0f} {MPa} $",
            rf"Vorspannung: $ \sigma_{{I}}(s_1) = {r['sigma_I_1']:.0f} {MPa} \leq -600 {MPa} $ → {'✔' if r['preload_ok'] else '✘ (Vorspannweg erhöhen, s_1 = 0,15 ... 0,2 h_0)'}",
        ]
        if sigma_H is not None:
            lines.append(
                rf"Dauerhubfestigkeit (TB 10-12): $ \sigma_h = {r['sigma_h']:.0f} \leq \sigma_H = {sigma_H:.0f} {MPa} $ → {'✔' if r['ok'] else '✘'}")
        else:
            lines.append(
                "Dauerhubfestigkeit $ \\sigma_H = \\sigma_O - \\sigma_U $ aus TB 10-12 ablesen und `sigma_H` übergeben.")
        return "\n\n".join(lines)


class DiscSpringStack(object):
    def __init__(self, spring: DiscSpring, n: int = 1, i: int = 1, wM: float = 0.0, wR: float = 0.0):
        """
        Federpaket / Federsäule aus gleichen Tellern.

        Args:
            spring (DiscSpring): Einzelteller
            n (int): Anzahl gleichsinnig geschichteter Teller je Paket
            i (int): Anzahl wechselsinnig aneinandergereihter Pakete (bei n = 1: Teller)
            wM, wR (float): Reibungsfaktoren für Mantel- und Randreibung (TB 10-13, z.B. 0,02 und 0,03)
        """
        self.spring, self.n, self.i, self.wM, self.wR = spring, n, i, wM, wR

    def friction_factor(self, loading: bool = True) -> float:
        """Faktor (1 -/+ wM (n-1) -/+ wR) aus Gl. 10.32; Belastung: minus, Entlastung: plus."""
        k = self.wM * (self.n - 1) + self.wR
        return 1 - k if loading else 1 + k

    def total_force(self, s_single: float, friction: bool = False, loading: bool = True) -> float:
        """Gesamtkraft F_ges = n F (Gl. 10.22), mit Reibung nach Gl. 10.32."""
        F = self.n * self.spring.force(s_single)
        return F / self.friction_factor(loading) if friction else F

    def calc_for_force(self, F_ges: float, s_required: float | None = None, friction: bool = False):
        """
        Auslegung für eine Gesamtkraft.

        Args:
            F_ges (float): Gesamtkraft [N]
            s_required (float, optional): geforderter Gesamtfederweg [mm]; dann wird i aufgerundet gewählt
                (Toleranz 0,01 Pakete, damit z.B. 12,0006 nicht zu 13 wird)
            friction (bool): Reibung bei Belastung nach Gl. 10.32 berücksichtigen
        """
        sp, n = self.spring, self.n
        self.F_ges = F_ges
        self.friction = friction
        self.F_single = F_ges * (self.friction_factor(True) if friction else 1.0) / n
        self.s_single = sp.travel_for_force(self.F_single)
        if s_required is not None:
            self.s_required = s_required
            self.i = np.ceil(s_required / self.s_single - 0.01)  # 0,01 Teller Toleranz gegen Rundungsgrenzfälle
        self.s_ges = self.i * self.s_single  # Gl. 10.23/10.24
        self.L0 = self.i * (sp.l0 + (n - 1) * sp.t)
        self.L = self.L0 - self.s_ges
        self.F_unloading = self.total_force(self.s_single, friction=True, loading=False) if friction else None
        return self.F_single, self.s_single, self.s_ges, self.L0, self.L

    def markdown(self) -> str:
        """Federpaket/Federsäule (nach calc_for_force())."""
        sp, n, i = self.spring, self.n, self.i
        N, mm = r"\, \text{N}", r"\, \text{mm}"
        lines = [
            "### Federpaket / Federsäule",
            "---",
            rf"Aufbau: $ n = {n} $ Teller gleichsinnig je Paket, $ i = {i} $ Pakete wechselsinnig; Teller $ D_e = {sp.De:g} {mm} $, $ t = {sp.t:g} {mm} $, $ l_0 = {sp.l0:g} {mm} $",
            rf"Gesamtkraft (Gl. 10.22): $ F_{{ges}} = n \, F = {_f(self.F_ges)} {N} $",
        ]
        if self.friction:
            fx = self.friction_factor(True)
            lines += [
                rf"Reibung (Gl. 10.32, Belastung): $ F = \frac{{F_{{ges,R}}}}{{n}} \left[ 1 - w_M (n-1) - w_R \right] = \frac{{{_f(self.F_ges)}}}{{{n}}} \cdot \left[ 1 - {self.wM:g} \cdot {n - 1} - {self.wR:g} \right] = {_f(self.F_single, 1)} {N} $ "
                rf"(Faktor ${fx:.3f}$)",
                rf"Bei Entlastung steht nur $ F_{{ges}} = {_f(self.F_unloading)} {N} $ zur Verfügung (Faktor $ 1 + w_M (n-1) + w_R $).",
            ]
        else:
            lines.append(
                rf"Kraft je Teller (ohne Reibung): $ F = \frac{{F_{{ges}}}}{{n}} = {_f(self.F_single, 1)} {N} $")
        lines += [
            rf"Federweg je Teller aus der Kennlinie (Gl. 10.25): $ s = {self.s_single:.3f} {mm} \; (s/h_0 = {self.s_single / sp.h0:.2f}) $"
            + (rf" ≤ $ s_{{0,75}} = {sp.s075:.3f} {mm} $ ✔" if self.s_single <= sp.s075 else ""),
            (
                rf"Anzahl der Pakete: $ i = \frac{{s_{{ges}}}}{{s}} = \frac{{{self.s_required:g}}}{{{self.s_single:.3f}}} \rightarrow i = {i} $"
                if hasattr(self, "s_required") else rf"Anzahl der Pakete: $ i = {i} $"),
            rf"Gesamtfederweg (Gl. 10.24): $ s_{{ges}} = i \, s = {i} \cdot {self.s_single:.3f} = {self.s_ges:.2f} {mm} $",
            rf"Länge unbelastet: $ L_0 = i \left[ l_0 + (n-1) t \right] = {i} \cdot \left[ {sp.l0:g} + {n - 1} \cdot {sp.t:g} \right] = {self.L0:.2f} {mm} $",
            rf"Länge belastet: $ L = L_0 - s_{{ges}} = {self.L:.2f} {mm} $",
        ]
        return "\n\n".join(lines)

