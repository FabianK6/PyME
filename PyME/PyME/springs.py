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

