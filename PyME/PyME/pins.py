#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bolzen/Stifte und Pinverbindungen nach Roloff/Matek Kap. 9 (Gln. 9.1 ... 9.4, 9.8, 9.18, 9.19, TB 9-1).

Nachweiskonzept des Buches (Maschinenbau): vorhandene Spannung <= zulässige Spannung
    sigma_b <= sigma_b,zul   (0,3 / 0,2 / 0,15 * Rm bei ruhender / schwellender / wechselnder Belastung)
    tau_a   <= tau_a,zul     (0,2 / 0,15 / 0,1  * Rm)
    p       <= p_zul         (0,35 / 0,25 * Rm des schwächeren Werkstoffs, nicht gleitende Flächen)
Kerbstifte: 0,8-fache Werte für sigma_b,zul und tau_a,zul, 0,7-fache für p_zul.
Der Anwendungsfaktor K_A (TB 3-4c) wird auf die Belastung angewendet.

Ablauf:
>>> pin = Pin(diameter=8, pinLength=22, insertionDept=12, material=steel, notched=True)
>>> conn = PinConnection(pin, part1=[12, 200], part2=[10, 400], num_of_pins=1)
>>> conn.calc_lateral_force(K_A=1.0, force=300, loadCase="pulsating")
>>> display(Markdown(conn.markdown()))
"""
import numpy as np
from materials import Solid
from misc import _f

# Faktoren auf R_m (Gl. 9.1 bis 9.4)
_LOADCASES_ = {
    "static":      dict(sigma_b=0.30, tau_a=0.20, p=0.35),
    "pulsating":   dict(sigma_b=0.20, tau_a=0.15, p=0.25),
    "alternating": dict(sigma_b=0.15, tau_a=0.10, p=None),   # p_zul für wechselnd steht nicht im Buch
}
_ALIASES_ = {"ruhend": "static", "schwellend": "pulsating", "wechselnd": "alternating"}


def _case(loadCase: str) -> str:
    c = _ALIASES_.get(loadCase.lower(), loadCase.lower())
    if c not in _LOADCASES_:
        raise ValueError("loadCase: 'static', 'pulsating' oder 'alternating' (ruhend/schwellend/wechselnd)")
    return c


class Pin(object):
    def __init__(self,
        diameter: float,
        pinLength: float,
        insertionDept: float,
        material: Solid,
        notched: bool = False
    ):
        """
        Stahlpin für Verbindungen oder Positionierungen von Bauteilen.

        Parameters:
            diameter (float): Pindurchmesser d [mm]
            pinLength (float): Pinlänge l [mm]
            insertionDept (float): Einstecktiefe im Bauteil mit Pressverbindung s [mm]
            material (Solid): Material des Pins (benötigt .Rm; für unbehandelte Normstifte Rm = 400 N/mm²)
            notched (bool): Kerbstift (Faktoren 0,8 / 0,8 / 0,7 auf die zulässigen Werte)
        """
        self.diameter = diameter
        self.pinLength = pinLength
        self.insertionDept = insertionDept
        self.material = material
        self.notched = notched
        self._update()

    def _update(self):
        """Abgeleitete Grössen aus dem Durchmesser (nach jeder Änderung von diameter aufrufen)."""
        d = self.diameter
        self.resistance = d ** 3 * np.pi / 32       # W für Biegung, Gl. 9.10b (das Buch rechnet auch mit 0,1 d³)
        self.crossSection = d ** 2 * np.pi / 4
        # Abschertragfähigkeit je Scherfläche nach Gl. 9.8 (Stahlbau, EN 1993-1-8): mit Bemessungswerten verwenden
        self.maxShearForce = 0.6 * self.crossSection * self.material.Rm / 1.25

    def allowable(self, loadCase: str = "pulsating"):
        """Zulässige Biege- und Schubspannung (sigma_b,zul, tau_a,zul) in N/mm² nach Gl. 9.1 / 9.3."""
        f = _LOADCASES_[_case(loadCase)]
        k = 0.8 if self.notched else 1.0
        return f["sigma_b"] * k * self.material.Rm, f["tau_a"] * k * self.material.Rm

    def rough_dimensioning(
        self,
        clampingFactor,
        allowedStress = None,
        force = None,
        K_A = 1.5,
        loadCase = "pulsating"
    ):
        """
        Überschlägige Durchmesserbestimmung eines Bolzens nach Roloff/Matek Gl. 9.1:
            d >= k * sqrt(K_A * F_nenn / sigma_b,zul)

        Einspannfaktor k (Klammerwerte bei Gleitverbindungen):
            1,6 (1,9) Einbaufall 1 (Bolzen lose in Stange und Gabel)
            1,1 (1,4) Einbaufall 2 (Bolzen mit Übermaßpassung in der Gabel)
            1,1 (1,2) Einbaufall 3 (Bolzen mit Übermaßpassung in der Stange)

        Args:
            clampingFactor (float): Einspannfaktor k
            allowedStress (float): zulässige BIEGESPANNUNG sigma_b,zul [N/mm²]. Nicht Rm!
                Gilt: 0,3 Rm (ruhend), 0,2 Rm (schwellend), 0,15 Rm (wechselnd); Normstift Rm = 400 N/mm².
                Ohne Angabe wird sie aus loadCase und material.Rm berechnet.
            force (float): Stangenkraft F_nenn [N]
            K_A (float): Anwendungsfaktor nach RM TB 3-4c
            loadCase (str): "static", "pulsating", "alternating" (nur wenn allowedStress None)

        Returns:
            float: Durchmesser d [mm] (wird auch in self.diameter gespeichert)
        """
        if allowedStress is None:
            allowedStress = self.allowable(loadCase)[0]
        self.diameter = clampingFactor * np.sqrt(K_A * force / allowedStress)
        self._update()
        return self.diameter

    def rough_dimensioning_markdown(self, clampingFactor, allowedStress=None, force=None, K_A=1.5, loadCase="pulsating") -> str:
        """Wie rough_dimensioning(), liefert aber die Rechnung als Markdown."""
        if allowedStress is None:
            allowedStress = self.allowable(loadCase)[0]
        d = self.rough_dimensioning(clampingFactor, allowedStress, force, K_A)
        lines = [
            "### Überschlägige Durchmesserbestimmung (Gl. 9.1)",
            "---",
            rf"Einspannfaktor: $ k = {clampingFactor} $, Anwendungsfaktor: $ K_A = {K_A} $, Stangenkraft: $ F_{{nenn}} = {_f(force)} \, \text{{N}} $",
            rf"Zulässige Biegespannung: $ \sigma_{{b,zul}} = {allowedStress:g} \, \frac{{\text{{N}}}}{{\text{{mm}}^2}} $",
            rf"Bolzendurchmesser: $ d \geq k \sqrt{{\frac{{K_A \, F_{{nenn}}}}{{\sigma_{{b,zul}}}}}} = {clampingFactor} \cdot \sqrt{{\frac{{{K_A} \cdot {_f(force)}}}{{{allowedStress:g}}}}} = {d:.2f} \, \text{{mm}} $",
            "Nächsten genormten Durchmesser wählen (TB 9-2, TB 9-3) und ein neues `Pin`-Objekt erstellen.",
        ]
        return "\n\n".join(lines)

    def markdown(self) -> str:
        """Kenngrössen des Pins."""
        d = self.diameter
        lines = [
            f"### Pin {d:g} x {self.pinLength:g}" + (" (Kerbstift)" if self.notched else ""),
            "---",
            rf"Durchmesser: $ d = {d:g} \, \text{{mm}} $, Länge: $ l = {self.pinLength:g} \, \text{{mm}} $, Einstecktiefe: $ s = {self.insertionDept:g} \, \text{{mm}} $",
            rf"Querschnittsfläche: $ A = \frac{{\pi d^2}}{{4}} = {self.crossSection:.2f} \, \text{{mm}}^2 $",
            rf"Widerstandsmoment: $ W = \frac{{\pi d^3}}{{32}} = {self.resistance:.2f} \, \text{{mm}}^3 $",
            rf"Zugfestigkeit: $ R_m = {self.material.Rm:g} \, \frac{{\text{{N}}}}{{\text{{mm}}^2}} $",
            rf"Abschertragfähigkeit je Scherfläche (Gl. 9.8, Stahlbau): $ F_{{v,Rd}} = \frac{{0.6 \, A \, R_m}}{{\gamma_{{M2}}}} = {_f(self.maxShearForce)} \, \text{{N}} $ mit $ \gamma_{{M2}} = 1.25 $",
        ]
        return "\n\n".join(lines)


class PinConnection(object):
    def __init__(
        self,
        pin: Pin,
        part1: list,
        part2: list,
        num_of_pins: int = 1,
        pressed_part: int = 1,
        shear_planes: int = 1,
    ):
        """
        Berechnung einer Pinverbindung zwischen zwei Werkstücken (Pin wie Steckstift, Gl. 9.18/9.19).

        Anwendungen:
            1: Drehmomentübertragung mit exzentrisch angeordneten Pins an einem Flansch
            2: Positionierung zweier Werkstücke wobei auf eines eine seitliche Kraft wirkt

        Args:
            pin (Pin): Pin-Objekt
            part1 (list): erstes Werkstück [Dicke t_1 (mm), Zugfestigkeit R_m (N/mm²)]
            part2 (list): zweites Werkstück [Dicke t_2 (mm), Zugfestigkeit R_m (N/mm²)]
            num_of_pins (int): Anzahl Pins in der Verbindung n
            pressed_part (int): 1 oder 2: Bauteil, in dem der Pin mit Presssitz (Einstecktiefe s) sitzt
            shear_planes (int): Anzahl Scherflächen je Pin (Flansch-Pins: 1; Gabelbolzen: 2)
        """
        self.pin = pin
        self.num_of_pins = num_of_pins
        self.part1 = part1
        self.part2 = part2
        self.pressed_part = pressed_part
        self.shear_planes = shear_planes

    # ------------------------------------------------------------------ Nachweise
    def _p_zul(self, partRm, loadCase):
        f = _LOADCASES_[loadCase]["p"]
        if f is None:
            raise ValueError("p_zul für wechselnde Belastung steht nicht im Buch: p_zul angeben (z.B. TB 9-1).")
        k = 0.7 if self.pin.notched else 1.0
        return f * k * min(self.pin.material.Rm, partRm)

    def _run(self, F_Q, loadCase, lever, p_zul):
        pin = self.pin
        loadCase = _case(loadCase)
        self.loadCase = loadCase
        self.force = F_Q
        self.lever = (pin.pinLength - pin.insertionDept) if lever is None else lever
        assert self.lever > 0, "Hebelarm l - s muss > 0 sein."
        d, s = pin.diameter, pin.insertionDept

        # zulässige Werte
        self.sigma_b_zul, self.tau_a_zul = pin.allowable(loadCase)
        pressedRm = (self.part1 if self.pressed_part == 1 else self.part2)[1]
        self.p1_zul = p_zul if p_zul is not None else self._p_zul(self.part1[1], loadCase)
        self.p2_zul = p_zul if p_zul is not None else self._p_zul(self.part2[1], loadCase)
        self.pmax_zul = p_zul if p_zul is not None else self._p_zul(pressedRm, loadCase)

        # vorhandene Werte (K_A steckt bereits in F_Q)
        self.bend = F_Q * self.lever / pin.resistance                               # Gl. 9.18
        self.shear = 4 / 3 * F_Q / (pin.crossSection * self.shear_planes)           # Gl. 9.3
        self.pressure1 = F_Q / (d * self.part1[0])                                  # Gl. 9.4
        self.pressure2 = F_Q / (d * self.part2[0])
        self.pressureMax = F_Q * (6 * self.lever + 4 * s) / (d * s ** 2) if s > 0 else float("nan")   # Gl. 9.19

        # Auslastung als Sicherheit gegen den zulässigen Wert (>= 1 erforderlich)
        self.SF_b = self.sigma_b_zul / self.bend
        self.SF_a = self.tau_a_zul / self.shear
        self.SF_1 = self.p1_zul / self.pressure1
        self.SF_2 = self.p2_zul / self.pressure2
        self.SF_pmax = self.pmax_zul / self.pressureMax
        return ([self.bend, self.shear, self.pressure1, self.pressure2, self.pressureMax],
                [self.SF_b, self.SF_a, self.SF_1, self.SF_2, self.SF_pmax])

    def calc_torque_transition(
            self, bore_set_diameter: float | int,
            K_A: float | int, torque: float | int,
            loadCase: str = "pulsating", lever: float | None = None, p_zul: float | None = None):
        """
        Drehmomentübertragung über n Pins auf dem Lochkreis.

        Parameters:
            bore_set_diameter (float|int): Lochkreisdurchmesser D [mm]
            K_A (float): Anwendungsfaktor nach RM TB 3-4c
            torque (float): Zu übertragendes Nenndrehmoment T [Nmm]
            loadCase (str): "static", "pulsating", "alternating"
            lever (float, optional): Hebelarm der Querkraft [mm]; Default l - s (Kraft am Pinende, konservativ)
            p_zul (float, optional): zulässige Flächenpressung [N/mm²] (z.B. aus TB 9-1), überschreibt Gl. 9.4-Faktoren

        Returns:
            (tuple): ([sigma_b, tau_max, p_1, p_2, p_max], [S_b, S_a, S_1, S_2, S_pmax], markdown_text)
        """
        self.mode = "torque"
        self.boreSetDiameter, self.K_A, self.torque = bore_set_diameter, K_A, torque
        self.torque_eq = torque * K_A
        F_Q = 2 * self.torque_eq / (bore_set_diameter * self.num_of_pins)     # F_Q = T_eq / (n * D/2)
        stresses, safeties = self._run(F_Q, loadCase, lever, p_zul)
        return stresses, safeties, self.markdown()

    def calc_lateral_force(
            self, K_A: float | int, force: float | int,
            loadCase: str = "pulsating", lever: float | None = None, p_zul: float | None = None):
        """
        Positionierung: seitliche Kraft F auf n Pins (gleichmässig verteilt).

        Returns:
            (tuple): ([sigma_b, tau_max, p_1, p_2, p_max], [S_b, S_a, S_1, S_2, S_pmax], markdown_text)
        """
        self.mode = "lateral"
        self.K_A, self.lateralForce = K_A, force
        self.force_eq = K_A * force
        F_Q = self.force_eq / self.num_of_pins
        stresses, safeties = self._run(F_Q, loadCase, lever, p_zul)
        return stresses, safeties, self.markdown()

    # ------------------------------------------------------------------ Markdown
    def markdown(self) -> str:
        """Rechnung als Markdown (nach calc_torque_transition() oder calc_lateral_force())."""
        pin = self.pin
        d, l, s = pin.diameter, pin.pinLength, pin.insertionDept
        t1, Rm1 = self.part1
        t2, Rm2 = self.part2
        n = self.num_of_pins
        MPa = r" \, \frac{\text{N}}{\text{mm}^2}"
        case_txt = {"static": "ruhend", "pulsating": "schwellend", "alternating": "wechselnd"}[self.loadCase]
        fac = _LOADCASES_[self.loadCase]
        kn = 0.8 if pin.notched else 1.0

        if self.mode == "torque":
            load = [
                rf"Äquivalentes Drehmoment: $ T_{{eq}} = K_A \, T = {self.K_A} \cdot {_f(self.torque)} \, \text{{Nmm}} = {_f(self.torque_eq)} \, \text{{Nmm}} $",
                rf"Querkraft pro Pin: $ F_Q = \frac{{T_{{eq}}}}{{n \cdot D/2}} = \frac{{2 \cdot {_f(self.torque_eq)}}}{{{n} \cdot {self.boreSetDiameter:g}}} = {_f(self.force, 1)} \, \text{{N}} $",
            ]
        else:
            load = [
                rf"Äquivalente Kraft: $ F_{{eq}} = K_A \, F = {self.K_A} \cdot {_f(self.lateralForce)} \, \text{{N}} = {_f(self.force_eq)} \, \text{{N}} $",
                rf"Querkraft pro Pin: $ F_Q = \frac{{F_{{eq}}}}{{n}} = \frac{{{_f(self.force_eq)}}}{{{n}}} = {_f(self.force, 1)} \, \text{{N}} $",
            ]

        def ok(S):
            return "✔" if S >= 1 else "✘"

        pf = fac["p"]
        pk = 0.7 if pin.notched else 1.0
        if pf is None:
            p_txt = r"$ p_{zul} $ vorgegeben"
        else:
            p_txt = rf"$ p_{{zul}} = {pk * pf:g} \, R_m $ des schwächeren Werkstoffs"

        lines = [
            "### Pinverbindung",
            "---",
            rf"Pin: $ d \times l = {d:g} \times {l:g} \, \text{{mm}} $, Einstecktiefe $ s = {s:g} \, \text{{mm}} $, Anzahl $ n = {n} $, "
            rf"Scherflächen je Pin: {self.shear_planes}" + (", Kerbstift" if pin.notched else ""),
            f"Belastung: **{case_txt}**",

            "#### Belastung",
            *load,
            rf"Hebelarm der Querkraft: $ l_h = l - s = {l:g} - {s:g} = {self.lever:g} \, \text{{mm}} $",

            "#### Zulässige Werte (Gln. 9.1, 9.3, 9.4)",
            rf"$ \sigma_{{b,zul}} = {fac['sigma_b'] * kn:g} \, R_m = {pin.allowable(self.loadCase)[0]:g} {MPa}, \quad "
            rf"\tau_{{a,zul}} = {fac['tau_a'] * kn:g} \, R_m = {self.tau_a_zul:g} {MPa} $",
            rf"Flächenpressung: {p_txt}; Bauteil 1: $ {self.p1_zul:g} {MPa} $, Bauteil 2: $ {self.p2_zul:g} {MPa} $",

            "#### Beanspruchungen und Nachweise",
            rf"Biegespannung (Gl. 9.18): $ \sigma_b = \frac{{F_Q \, l_h}}{{W}} = \frac{{{_f(self.force, 1)} \cdot {self.lever:g}}}{{{pin.resistance:.2f}}} = {self.bend:.1f} {MPa} \leq {self.sigma_b_zul:.1f} {MPa} $ → {ok(self.SF_b)} ($ S = {self.SF_b:.2f} $)",
            rf"Grösste Schubspannung (Gl. 9.3): $ \tau_{{max}} = \frac{{4}}{{3}} \frac{{F_Q}}{{{self.shear_planes} \cdot A}} = {self.shear:.1f} {MPa} \leq {self.tau_a_zul:.1f} {MPa} $ → {ok(self.SF_a)} ($ S = {self.SF_a:.2f} $)",
            rf"Mittlere Flächenpressung Bauteil 1 (Gl. 9.4): $ p_1 = \frac{{F_Q}}{{d \, t_1}} = \frac{{{_f(self.force, 1)}}}{{{d:g} \cdot {t1:g}}} = {self.pressure1:.1f} {MPa} \leq {self.p1_zul:.1f} {MPa} $ → {ok(self.SF_1)}",
            rf"Mittlere Flächenpressung Bauteil 2 (Gl. 9.4): $ p_2 = \frac{{F_Q}}{{d \, t_2}} = \frac{{{_f(self.force, 1)}}}{{{d:g} \cdot {t2:g}}} = {self.pressure2:.1f} {MPa} \leq {self.p2_zul:.1f} {MPa} $ → {ok(self.SF_2)}",
            rf"Grösste Pressung in der Presssitzbohrung (Bauteil {self.pressed_part}, Gl. 9.19): $ p_{{max}} = \frac{{F_Q (6 \, l_h + 4 s)}}{{d \, s^2}} = \frac{{{_f(self.force, 1)} \cdot (6 \cdot {self.lever:g} + 4 \cdot {s:g})}}{{{d:g} \cdot {s:g}^2}} = {self.pressureMax:.1f} {MPa} \leq {self.pmax_zul:.1f} {MPa} $ → {ok(self.SF_pmax)}",
        ]
        return "\n\n".join(lines)