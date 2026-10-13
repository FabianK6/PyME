#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 23:07:52 2026

@author: fabian
"""

import numpy as np
from  misc import _e, _f


class Bearing(object):
    def __init__(self,
                 innerDiameter: float,
                 outerDiameter: float,
                 axialForce: float | np.ndarray,
                 radialForce: float | np.ndarray,
                 frequency: float | np.ndarray
                 ) -> None:
        """
        This Class is a Baseclass und does not need to be invoced by the user.
        Use RollerBearing or SlideBearing instead.
        """
        self.innerDiameter = innerDiameter
        self.outerDiameter = outerDiameter
        self.axialForce = axialForce
        self.radialForce = radialForce
        self.frequency = frequency
        pass


class RollerBearing(Bearing):
    def __init__(self,
        innerDiameter: float,
        outerDiameter: float,
        axialForce: float|np.ndarray,
        axialFactor: float,
        radialForce: float|np.ndarray,
        radialFactor: float,
        frequency: float|np.ndarray,
        bearingtype: str = "Ball"
    ):
        """
        This Class represents a mechanical Roller Bearing.

        Args:
            innerDiameter (float): [mm]
            outerDiameter (float): [mm]
            axialForce (float|np.ndarray): [N]
            axialFactor (float): X
            radialForce (float|np.ndarray): [N]
            radialFactor (float): Y
            frequency (float|np.ndarray): [rpm]
            bearingtype (str): "Ball", or "Roll"
        """
        super().__init__(
            innerDiameter,
            outerDiameter,
            axialForce,
            radialForce,
            frequency
            )
        self.axialFactor = axialFactor
        self.radialFactor = radialFactor
        match bearingtype.lower():
            case "ball":
                self.p = 3
            case "roll":
                self.p = 10/3
            
    def calc_with_constant_load(self, fL):
        """
        calculate bearing according to Roloff Matek CH14
        P = Fa x Y + Fr x X
        C = P x fL / fn
        fn = ((33+1/3)/f)^(1/p) where f = frequency and p = 3 if rollerbearing else 10/3
        L10 = (C/P)^p

        Args:
            fL (float): lifetime factor:
            >>> -------------- | störungen durch Lagerwechsel
            >>> Betriebsart--- | sehr gestört | weniger gestört
            >>> Aussetzbetrieb | fL = 2...3.5 | fL = 1...2.5
            >>> Zeitbetrieb--- | fL = 3...4.5 | fL = 2...4.0
            >>> Dauerbetrieb-- | fL = 4...5.5 | fL = 3.5...5
        """
        radial = self.radialForce * self.radialFactor
        axial = self.axialForce * self.axialFactor
        self.load = radial + axial
        
        fn = ((33+1/3)/self.frequency)**(1/self.p)
        self.carryNumber = self.load * fL / fn
        
        self.L10 = (self.carryNumber/self.load)**self.p
        self.L10h = 1e6 * self.L10 / (60 * self.frequency)
        
    def calc_with_loadprofile(self, fL, percentages):
        nm = np.sum(percentages * self.frequency)
        Parr = (self.radialForce * self.radialFactor + self.axialForce * self.axialFactor)**self.p
        self.load = np.sum(Parr * self.frequency / nm * percentages)**(1/self.p)
        fn = ((33+1/3)/nm)**(1/self.p)
        self.carryNumber = self.load * fL / fn
        self.L10 = (self.carryNumber/self.load)**self.p
        self.L10h = 1e6 * self.L10 / (60 * nm)

    def _typ(self):
        if self.p == 3:
            return "Kugellager", r"\frac{1}{3}"
        return "Rollenlager", r"\frac{3}{10}"

    def markdown_constant_load(self, fL) -> str:
        """Rechnet mit calc_with_constant_load() und gibt die Rechnung als Markdown zurück."""
        self.calc_with_constant_load(fL)
        typ, exp = self._typ()
        fn = ((33 + 1 / 3) / self.frequency) ** (1 / self.p)
        N = r"\, \text{N}"
        lines = [
            f"### Wälzlager ({typ}) – konstante Belastung",
            "---",
            rf"Lagerabmessungen: $ d = {self.innerDiameter} \, \text{{mm}}, \quad D = {self.outerDiameter} \, \text{{mm}} $",
            rf"Lebensdauerexponent: $ p = {self.p:.3g} $",

            "#### Belastung",
            rf"Äquivalente dynamische Lagerbelastung: "
            rf"$ P = X F_r + Y F_a = {self.radialFactor} \cdot {_f(self.radialForce)} {N} + {self.axialFactor} \cdot {_f(self.axialForce)} {N} = {_f(self.load)} {N} $",

            "#### Tragzahl",
            rf"Drehzahlfaktor: $ f_n = \left( \frac{{33.3}}{{n}} \right)^{{1/p}} = \left( \frac{{33.3}}{{{_f(self.frequency, 1)}}} \right)^{{{exp}}} = {_f(fn, 3)} $",
            rf"Lebensdauerfaktor: $ f_L = {fL} $",
            rf"Erforderliche dynamische Tragzahl: $ C_{{erf}} = P \frac{{f_L}}{{f_n}} = {_f(self.load)} \cdot \frac{{{fL}}}{{{_f(fn, 3)}}} = {_f(self.carryNumber)} {N} $",

            "#### Lebensdauer",
            rf"Nominelle Lebensdauer: $ L_{{10}} = \left( \frac{{C}}{{P}} \right)^p = {_f(self.L10, 1)} \cdot 10^6 \, \text{{Umdrehungen}} $",
            rf"Lebensdauer in Stunden: $ L_{{10h}} = \frac{{L_{{10}} \cdot 10^6}}{{60 \, n}} = {_f(self.L10h)} \, \text{{h}} $",
            "Es ist ein Lager zu wählen, dessen dynamische Tragzahl $ C \\geq C_{erf} $ ist.",
        ]
        return "\n\n".join(lines)

    def markdown_loadprofile(self, fL, percentages) -> str:
        """Rechnet mit calc_with_loadprofile() und gibt die Rechnung samt Lastkollektiv-Tabelle als Markdown zurück."""
        q = np.asarray(percentages, dtype=float)
        self.calc_with_loadprofile(fL, q)
        typ, exp = self._typ()
        n = np.atleast_1d(self.frequency * np.ones_like(q))
        Fr = np.atleast_1d(self.radialForce * np.ones_like(q))
        Fa = np.atleast_1d(self.axialForce * np.ones_like(q))
        P_i = Fr * self.radialFactor + Fa * self.axialFactor
        nm = np.sum(q * n)
        fn = ((33 + 1 / 3) / nm) ** (1 / self.p)
        N = r"\, \text{N}"

        table = [
            r"| $i$ | $q_i$ | $n_i$ | $F_{r,i}$ [N] | $F_{a,i}$ [N] | $P_i$ [N] |",
            "|:-:|:-:|:-:|:-:|:-:|:-:|",
        ]
        for i, (qi, ni, fr, fa, pi_) in enumerate(zip(q, n, Fr, Fa, P_i), start=1):
            table.append(f"| {i} | ${qi:g}$ | ${ni:g}$ | ${_f(fr)}$ | ${_f(fa)}$ | ${_f(pi_)}$ |")

        lines = [
            f"### Wälzlager ({typ}) – Lastkollektiv",
            "---",
            rf"Lagerabmessungen: $ d = {self.innerDiameter} \, \text{{mm}}, \quad D = {self.outerDiameter} \, \text{{mm}} $",
            rf"Lebensdauerexponent: $ p = {self.p:.3g} $",
            "#### Lastkollektiv",
            "\n".join(table),

            "#### Äquivalente Belastung und Tragzahl",
            rf"Mittlere Drehzahl: $ n_m = \sum_i q_i n_i = {_f(nm, 2)} $",
            rf"Äquivalente dynamische Lagerbelastung: $ P = \left( \sum_i P_i^p \frac{{n_i \, q_i}}{{n_m}} \right)^{{1/p}} = {_f(self.load)} {N} $",
            rf"Drehzahlfaktor: $ f_n = \left( \frac{{33.3}}{{n_m}} \right)^{{1/p}} = {_f(fn, 3)} $",
            rf"Erforderliche dynamische Tragzahl: $ C_{{erf}} = P \frac{{f_L}}{{f_n}} = {_f(self.load)} \cdot \frac{{{fL}}}{{{_f(fn, 3)}}} = {_f(self.carryNumber)} {N} $",

            "#### Lebensdauer",
            rf"Nominelle Lebensdauer: $ L_{{10}} = \left( \frac{{C}}{{P}} \right)^p = {_f(self.L10, 1)} \cdot 10^6 \, \text{{Umdrehungen}} $",
            rf"Lebensdauer in Stunden: $ L_{{10h}} = \frac{{L_{{10}} \cdot 10^6}}{{60 \, n_m}} = {_f(self.L10h)} \, \text{{h}} $",
        ]
        return "\n\n".join(lines)


class SlideBearing(Bearing):
    def __init__(self, 
        innerDiameter: float,
        outerDiameter: float,
        axialForce: float,
        radialForce: float,
        frequency: float,
        widthRatio: float,
        dynViscosity: float,
        relativeExcentricity: float,
        beta: float
    ):
        """
        This Class represents a hydromechanical slide bearing.

        Args:
            innerDiameter (float): [mm]
            outerDiameter (float): [mm]
            axialForce (float): [N]
            radialForce (float): [N]
            frequency (float): [Hz]
            widthRatio (float): width / innerDiameter [-]
            dynViscosity (float): [N*s/mm²]
            relativeExcentricity (float): [-]
            beta (float): angle between force and pressure max [rad]
        """
        super().__init__(
            innerDiameter, 
            outerDiameter, 
            axialForce, 
            radialForce,
            frequency
        )
        self.widthRatio = widthRatio
        self.dynViscosity = dynViscosity
        self.relativeExcentricity = relativeExcentricity
        self.beta = beta
        
    def calc_with_constant_load(self):
        self.width = self.widthRatio * self.innerDiameter
        self.pL = self.radialForce / (self.innerDiameter * self.width)
        
        self.uW = self.innerDiameter * 1e-3 * np.pi * self.frequency
        self.psi_B = 0.8 * self.uW**(1/4) * 1e-3
        
        dgr = [63, 160, 400, 1000, 2500]
        ugr = [1, 3, 10, 30, 3000]
        h0gr = np.array([
            [3, 4, 5, 7, 10],
            [4, 5, 7, 9, 12],
            [6, 7, 9, 11, 14],
            [8, 9, 11, 13, 16],
            [10, 12, 14, 16, 18]
            ]) / 1000
        
        for ind1, d in enumerate(dgr):
            if self.innerDiameter <= d:
                i = ind1
            else:
                i = len(dgr)-1

        for ind2, u in enumerate(ugr):
            if self.uW <= u:
                j = ind2
            else:
                j = len(ugr)-1
                
        self.h0min = h0gr[i, j]
        self.h0 = 0.5 * self.innerDiameter * self.psi_B * (1 - self.relativeExcentricity)
        
        self.omega = 2*np.pi*self.frequency
        
        d = (self.dynViscosity * self.omega)
        self.sommerfeld = self.pL * self.psi_B**2 / d
        
        m = np.pi / (self.sommerfeld * np.sqrt(1 - self.relativeExcentricity**2))
        n = self.relativeExcentricity/2 * np.sin(self.beta)
        self.friction = self.psi_B * (m + n)
        
    def flowrate(self, Vp_rel):
        a = self.widthRatio - 0.223 * self.widthRatio**3
        Vdrel = self.relativeExcentricity * 0.25 * a
        self.VD = Vdrel * self.innerDiameter**3 * self.psi_B * self.omega
        self.Vp = Vp_rel * self.innerDiameter**3 * self.psi_B**3 / self.dynViscosity
        return self.Vp + self.VD
    
    def temperature(self, flowrate, T_in, spezHeatCap, density):
        self.PR = self.friction * self.radialForce * self.uW
        return T_in + self.PR / (spezHeatCap * flowrate * density)
  