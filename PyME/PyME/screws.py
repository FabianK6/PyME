#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 23:20:11 2026

@author: fabian

Berechne eine Schraubenverbindung gemäss Roloff Matek Kapitel 8. \n
Eine Verbindung zwischen zwei Werkstücken durch Schrauben wird in drei Schritten ausgeführt: \n
Es muss eine provisorische Instanz eines Schrauben-Objekts erstellt werden. \n
>>> screw = Screw(...) \n
Danach kann mit der entsprechenden Methode ein mindest-Spannungsquerschnitt der Schraube berechnet werden. \n
>>> screw.provisorial_dimension(...) \n
Hat man eine Schraube gewählt, muss diese als neues Objekt instanziert werden. \n
>>> new_screw = Screw(...)
Nun kann eine Schraube mit einem Spannungsquerschnitt gewählt werden, der mindestens so gross ist, wie berechnet. \n
Quellen dazu sind Herstellerkataloge, Tabellenbücher und der Normenauszug. \n
Nun kann die Instanz der Schraubenverbindung erstellt werden. \n
>>> connection = ScrewConnection(...) \n
Nun kann nun die Methode zur Berechnung aufgerufen werden. \n
Hier wird in einem ersten Schritt die Längenänderungen berechnet und eine benötigte Schraubenkraft ermittelt.
>>> connection.precalculation(...)
Damit kann nun aus den Katalogdaten wiederum eine passende Schraube neu gewählt oder die maximale Schraubenkraft der bereits
gewählten Schraube herausgesucht werden. \n
Danach können die Sicherheiten und das Anzugsdrehmoment der Schraube ermittelt werden sowie ein Spannungsschaubild gezeichnet werden.
>>> connection.calculation(...) \n
"""
from materials import Solid
import numpy as np
from misc import _e, _f


_SCREWCLASSES_ = np.array(["4.6", "8.8", "10.9", "12.9"])
_SCREWSIZES_ = np.array(["M1.6", "M2", "M2.5", "M3", "M4", "M5", "M6", "M8", "M10", "M12", "M16", "M20", "M24", "M27", "M30", "M33", "M36"])
_PITCHTABLE_ = np.array([0.35, 0.4, 0.45, 0.5, 0.7, 0.8, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3, 3.5, 3.5, 4])
_SCREWTABLE_ = np.array([
    [228, 607, 854, 1025],
    [378, 756, 1008, 1417],
    [628, 1676, 2356, 2828],
    [941, 2510, 3530, 4236],
    [1630, 4400, 6500, 7600],
    [2670, 7200, 10600, 12400],
    [3760, 10200, 14900, 17500],
    [6900, 18600, 27300, 32000],
    [11000, 29600, 43400, 50800],
    [16000, 43000, 63200, 74000],
    [30100, 80900, 118800, 139000],
    [47000, 130000, 186000, 217000],
    [67700, 188000, 267000, 313000],
    [89000, 246000, 351000, 410000],
    [108300, 300000, 427000, 499000],
    [134700, 373000, 531000, 621000],
    [158200, 438000, 623000, 729000],
])

class WorkPiece(object):
    def __init__(self, thickness: float|int, material: Solid):
        self.thickness = thickness
        self.material = material


class Screw(object):
    def __init__(self,
        nominalDiameter: int,
        pitch: float,
        threadType: str,
        head: str,
        headDiameter: float,
        threadLength: float,
        shaftLength: float,
        toughnessClass: str,
        young: float = 210_000
    ):
        """
        Screw-type object

        Args:
            nominalDiameter (int): [mm]
            pitch (float): [mm]
            threadType (str): metric [M] or imperial [G]
            head (str): hexagonal [hex] or cylindrical [cyl]
            headDiameter (float): Contact surface diameter of the screwhead [mm]
            threadLength (float): [mm]
            shaftLength (float): [mm]
            toughnessClass (str): [4.6, 8.8, 10.2, 12.9]
            young (float, optional): [Mpa]. Defaults to 210_000.
        """
        self.nominalDiameter = nominalDiameter
        self.pitch = pitch
        self.threadType = threadType
        self.headDiameter = headDiameter
        self.head = head
        self.threadLength = threadLength
        self.shaftLength = shaftLength
        self.toughnessClass = toughnessClass
        self.young = young
        T, C = toughnessClass.split(".")
        self.Rp_02 = float(T) * float(C) * 10
        self.Rm = float(T) * 100
        self.length = self.threadLength + self.shaftLength
        if "M" in self.threadType.upper():
            self.flankOverlap = 0.5413*pitch
            self.flankAngle = np.pi / 3
            a = 0.6495
            b = 1.22687
        elif "G" in self.threadType.upper():
            self.flankOverlap = 0.64*pitch
            self.flankAngle = 55 / 180 * np.pi
            a = 0.6495
            b = 1.28
        self.flankDiameter = self.nominalDiameter - a * self.pitch
        self.coreDiameter = self.nominalDiameter - b * self.pitch
        self.strainCrossSection = (self.flankDiameter + self.coreDiameter)**2 * np.pi / 16
        self.strainDiameter = (self.flankDiameter + self.coreDiameter) / 2
        self.pitchAngle = np.arctan(self.pitch / (self. flankDiameter * np.pi))
        self.toleratedDynamicStress = 0.85 * (150 / self.nominalDiameter + 45)

    def provisorial_dimension(
            self, screwForce,
            clampingLength,
            KA, saturation = 0.9
        ) -> tuple:
        """
        calculate tension area and diameter needed.

        Args:
            scewForce (float): Force acting on the screw [N]
            clampingLength (float): distance between screw head and nut/threaded piece [m]
            KA (float): application factor
            saturation (float): percentage used of the Rp0.2 value of the screw, must be in range [0, 1]

        returns:
            tension-cross-section, tension-diameter

        """
        if self.shaftLength / self.length < 0.01:
            beta = 0.8
        else:
            beta = 1.1
        As = screwForce / (self.Rp_02 * saturation * KA - beta * self.young * 0.011 / clampingLength)
        ds = np.sqrt(As * 4 / np.pi)
        return As, ds

    def markdown(self) -> str:
        """Geometrie- und Festigkeitskennwerte der Schraube."""
        b = (self.nominalDiameter - self.coreDiameter) / self.pitch
        alpha_deg = np.degrees(self.pitchAngle)
        lines = [
            f"### Schraube {self.threadType.upper()}{self.nominalDiameter} x {self.pitch} - {self.toughnessClass}",
            "---",
            "#### Geometrie",
            rf"Flankendurchmesser: $ d_2 = d - 0.6495 \, P = {self.nominalDiameter} - 0.6495 \cdot {self.pitch} = {self.flankDiameter:.3f} \, \text{{mm}} $",
            rf"Kerndurchmesser: $ d_3 = d - {b:.5g} \, P = {self.nominalDiameter} - {b:.5g} \cdot {self.pitch} = {self.coreDiameter:.3f} \, \text{{mm}} $",
            rf"Spannungsdurchmesser: $ d_S = \frac{{d_2 + d_3}}{{2}} = {self.strainDiameter:.3f} \, \text{{mm}} $",
            rf"Spannungsquerschnitt: $ A_S = \frac{{\pi}}{{16}} (d_2 + d_3)^2 = {self.strainCrossSection:.2f} \, \text{{mm}}^2 $",
            rf"Steigungswinkel: $ \alpha = \arctan \left( \frac{{P}}{{\pi d_2}} \right) = {alpha_deg:.3f}^\circ $",
            "#### Festigkeit",
            rf"Festigkeitsklasse {self.toughnessClass}: $ R_m = {self.Rm:.0f} \, \frac{{\text{{N}}}}{{\text{{mm}}^2}}, \quad R_{{p0.2}} = {self.Rp_02:.0f} \, \frac{{\text{{N}}}}{{\text{{mm}}^2}} $",
            rf"Ausschlagfestigkeit: $ \sigma_{{ASV}} = 0.85 \left( \frac{{150}}{{d}} + 45 \right) = {self.toleratedDynamicStress:.1f} \, \frac{{\text{{N}}}}{{\text{{mm}}^2}} $",
        ]
        return "\n\n".join(lines)

    def provisorial_dimension_markdown(self, screwForce, clampingLength, KA, saturation=0.9) -> str:
        """Wie provisorial_dimension(), liefert aber die Rechnung als Markdown."""
        As, ds = self.provisorial_dimension(screwForce, clampingLength, KA, saturation)
        beta = 0.8 if self.shaftLength / self.length < 0.01 else 1.1
        lines = [
            "### Provisorische Dimensionierung",
            "---",
            rf"Beiwert: $ \beta = {beta} $ (Schaftlänge {'< 1 %' if beta == 0.8 else '≥ 1 %'} der Schraubenlänge)",
            rf"Mindest-Spannungsquerschnitt: "
            rf"$$ A_{{S,min}} = \frac{{F}}{{R_{{p0.2}} \, \kappa \, k_A - \beta \, E \, \frac{{0.011}}{{l_K}}}} "
            rf"= \frac{{{_f(screwForce)} }}{{{self.Rp_02:.0f} \cdot {saturation} \cdot {KA} - {beta} \cdot {_f(self.young)} \cdot \frac{{0.011}}{{{clampingLength}}}}} "
            rf"= {As:.2f} \, \text{{mm}}^2 $$",
            rf"Mindest-Spannungsdurchmesser: $ d_{{S,min}} = \sqrt{{\frac{{4 A_{{S,min}}}}{{\pi}}}} = {ds:.2f} \, \text{{mm}} $",
            "Eine Schraube wählen, deren Spannungsquerschnitt mindestens so gross ist.",
        ]
        return "\n\n".join(lines)


class ScrewConnection(object):
    def __init__(self,
        screw: Screw,
        clamping_piece: WorkPiece,
        threaded_piece: WorkPiece,
        outerDiameter: float|int,
        forceapplication: float|int,
        staticforce: float|int,
        appliedforce: float|int,
        dynamicForceAmplitude: float|int,
        fasteningstyle: float|int,
        threadfriction: float|int,
        piecefriction: float|int,
        safetyfactor: float|int,
    ):
        """
        Connection between a Screw, a threaded Piece and a clamped piece.
        Calculates the needed Torque to reach the allowed Stress within the bolt.

        Args:
            screw (Screw): Screw-Object
            clamping_piece (WorkPiece): clamped piece
            threaded_piece (WorkPiece): threaded piece or nut
            forceapplication (float): takes the Setup pieces into account (n)
            fasteningforce (float): F_V
            appliedforce (float): F_B
            dynamicForceAmplitude (float): F_a
            fasteningstyle (float): usually 1.4 if using a wrench
            threadfriction (float): friction within thread, usually 0.12 for steel on steel
            piecefriction (float): friction between pieces, usually 0.12 for steel on steel
            safetyfactor (float): Safety against breaking the connection
        """
        self.screw = screw
        self.clamping_piece = clamping_piece
        self.threaded_piece = threaded_piece
        self.outerDiameter = outerDiameter
        self.forceapplication = forceapplication
        self.staticforce = staticforce
        self.appliedforce = appliedforce
        self.dynamicForceAmplitude = dynamicForceAmplitude
        self.fasteningstyle = fasteningstyle
        self.threadfriction = threadfriction
        self.piecefriction = piecefriction
        self.safetyfactor = safetyfactor

    def precalculation(self) -> tuple:
        bore_diameter = (self.screw.nominalDiameter + 1)
        ### Calculating elongation of screw ###
        if self.screw.head.lower() == "hex":
            l_ko = 0.5 * self.screw.nominalDiameter
        else:
            l_ko = 0.4 * self.screw.nominalDiameter

        l_Ge = 0.5 * self.screw.nominalDiameter

        AN = self.screw.nominalDiameter ** 2 / 4 * np.pi
        A3 = self.screw.coreDiameter ** 2 * np.pi / 4
        lG = self.clamping_piece[0] - self.screw.shaftLength
        a = ((self.screw.shaftLength + l_ko) / AN + (l_Ge + lG) / A3)
        b = 0.4 * self.screw.nominalDiameter / (self.threaded_piece[0].young * AN)
        delta_s = 1 / self.screw.young * a + b

        ### Calculation of elongation of Pieces ###
        logic1 = self.screw.headDiameter <= self.outerDiameter
        logic2 = self.outerDiameter <= self.screw.headDiameter + self.clamping_piece.thickness
        logic3 = self.outerDiameter < self.outerDiameter
        if logic1 or logic2:
            s1 = (self.screw.headDiameter ** 2 - bore_diameter ** 2)
            s2 = (self.outerDiameter - self.screw.headDiameter)
            x = (
                            self.clamping_piece.thickness * self.screw.headDiameter / self.outerDiameter ** 2) ** (
                            1 / 3)
            A_ers = np.pi / 4 * s1 + np.pi / 8 * self.screw.headDiameter * s2 * ((x + 1) ** 2 - 1)
        elif logic3:
            A_ers = np.pi * (self.outerDiameter ** 2 - bore_diameter ** 2) / 4
        try:
            delta_t = self.clamping_piece.thickness / (A_ers * self.threaded_piece.material.young)
        except Exception:
            delta_t = 0
        ### Ermitteln der benötigten Schraube
        compressionSetValue = 3.29 * (self.clamping_piece.thickness / self.screw.nominalDiameter) ** 0.34 * 10 ** (-3)
        powerfactor = self.forceapplication * delta_t / (delta_s + delta_t)
        clampingforce = self.staticforce - self.appliedforce * (1 - powerfactor)
        stickforce = compressionSetValue / (delta_s + delta_t)
        neededAssemblyForce = self.fasteningstyle * (clampingforce + self.appliedforce * (1 - powerfactor) + stickforce)

        mask = _SCREWTABLE_ >= neededAssemblyForce
        rows, cols = np.where(mask)

        pairs = [
            f"{_SCREWSIZES_[r]} x {_PITCHTABLE_[r]} - {_SCREWCLASSES_[c]}, FS = {_SCREWTABLE_[r, c]} N"
            for r, c in zip(rows, cols)
        ]
        return powerfactor, clampingforce, stickforce, neededAssemblyForce, compressionSetValue, delta_s, delta_t, pairs

    def calculation(
            self, powerfactor, clampingforce, stickforce, assemblyforce, compressionSetValue, delta_s, delta_t
    ):
        """
        Calculate Screwconnection taking screw elongation and workpiece squashing into account.

        calculates the following attributes:
            delta_s (float): Elongation of the screw δ_s [1]
            delta_t (float): retraction of the clamped piece δ_t [1]
            powerfactor (float): Φn [1]
            clampingforce (float): F_Kl [N]
            stickforce (float): F_Z [N]
            assemblyforce (float): F_VM [N]
            threadtorquep (float): torque in thread when losening M_G+ [Nmm]
            threadtorquen (float): torque in thread when fastening M_G- [Nmm]
            fasteningtorque (float): assembly torque of the screw M_A [Nmm]
            sigma_M (float): assemblystress σ_M [MPa]
            tauTp (float): screw polar strain when losening [MPa]
            tauTn (float): screw polar strain when fastening [MPa]
            sigma_redp (float): reduced screwstress when losening [MPa]
            sigma_redn (float): reduced screwstress when fastening [MPa]
        """
        self.delta_s = delta_s
        self.delta_t = delta_t
        self.powerfactor = powerfactor
        self.clampingforce = clampingforce
        self.assemblyforce = assemblyforce
        self.stickforce = stickforce
        self.compressionSetValue = compressionSetValue

        self.screwHeadArea = (self.screw.headDiameter**2 - (self.screw.nominalDiameter + 1)**2) * np.pi / 4
        Rpzul = self.screw.Rp_02 * 0.9
        self.sigma_M = Rpzul / (1 + 3 * (3 / self.screw.nominalDiameter * (0.159 * self.screw.pitch + 0.577 * self.threadfriction * self.screw.flankDiameter)**2))**(1 / 2)

        ### Kräfte ###
        self.dynamicMeanForce = self.dynamicForceAmplitude * self.powerfactor + self.staticforce
        self.allowedScrewForce = 0.1 * self.screw.Rp_02 * self.screw.strainCrossSection
        self.additionalForce = self.powerfactor * self.appliedforce
        self.maxPullForce = self.assemblyforce + self.additionalForce

        ### Anzugsmomente ###
        rho = self.threadfriction / np.cos(self.screw.flankAngle / 2)
        self.threadtorquep = self.assemblyforce * self.screw.strainDiameter / 2 * np.tan(self.screw.pitchAngle + rho)
        self.threadtorquen = self.assemblyforce * self.screw.strainDiameter / 2 * np.tan(self.screw.pitchAngle - rho)
        self.fasteningtorque = self.assemblyforce * (self.screw.strainDiameter / 2 * np.tan(
            self.screw.pitchAngle + rho) + self.piecefriction * self.screw.headDiameter)

        ### Spannungen ###
        Wt = np.pi * self.screw.strainDiameter ** 3 / 12
        self.tauTn = self.threadtorquen / Wt
        self.tauTp = self.threadtorquep / Wt
        self.sigma_zmax = self.maxPullForce / self.screw.strainCrossSection
        self.sigma_redp = (self.sigma_zmax ** 2 + 3 * self.tauTp ** 2) ** (1 / 2)
        self.sigma_redn = (self.sigma_zmax ** 2 + 3 * self.tauTn ** 2) ** (1 / 2)
        self.dynamicStress = self.dynamicMeanForce / self.screw.strainCrossSection

        ### Sicherheitsberechnungen ###
        self.staticSafetyp = self.screw.Rp_02 / self.sigma_redp
        self.staticSafetyn = self.screw.Rp_02 / self.sigma_redn
        self.forceSafety = self.allowedScrewForce / self.additionalForce
        self.areaPressure = (self.assemblyforce + self.powerfactor * self.appliedforce) / self.screwHeadArea
        self.areaSafety = self.clamping_piece.material.Re / self.areaPressure
        self.dynamicSafety = self.screw.toleratedDynamicStress / self.dynamicStress
        return

    def complex_markdown(self):
        return (
            " ### Schraubenverbindung " " \n " " *** " " \n "
            f" Empfohlene Schraube: {self.screw.threadType + str(self.screw.nominalDiameter)} x {self.screw.shaftLength + self.screw.threadLength} - {self.screw.toughnessClass}"
            r" Für Ergebnisse mit zwei Angaben gilt: $ [tan(\alpha + \rho), tan(\alpha - \rho)] $ weil dort das Gewindemoment in die Berechnung einfliesst. " " \n "
            " #### Kräfte " " \n "
            r" Klemmkraft der Schraube: $ F_{Kl} = F_v - F_B (1 - \Theta) = " f" {int(self.clampingforce)} N " " $ " " \n "
            r" Setzkraft der Schraube: $ F_z = \frac{0.011}{\delta_S + \delta_T} = " f" {int(self.stickforce)} N " " $ " " \n "
            r" Montagekraft der Schraube: $ F_{MV} = k_A (F_{Kl} + F_B (1 - \Theta) + F_z) = " f" {int(self.assemblyforce)} N " " $ " " \n "
            r" Zusatzkraft: $ F_{SB} = F_B \Theta = " f" {int(self.additionalForce)} N $ " " \n "
            r" maximal zulässige Schraubenkraft: $ F_{S,max} = 0.1 R_{p0.2} A_s = " f" {int(self.allowedScrewForce)} N $ " " \n "
            " #### Gewinde- und Anzugsmoment " " \n "
            r" Gewindemoment: $ M_G = F_{MV} \frac{d_2}{2} tan(\alpha \pm \rho) = " f" [{int(self.threadtorquep)}, {int(self.threadtorquen)}] Nmm " " $ " " \n "
            r" Anzugsmoment: $ M_A = F_{MV} (\frac{d_2}{2} tan(\alpha + \rho) + \mu d_K) = " f" {int(self.fasteningtorque)} Nmm " " $ " " \n "
            " #### Spannungen und Dehnungen " " \n "
            r" Dehnung der Schraube: $ \delta_S = " f" {self.delta_s * self.staticforce} $ " " \n "
            r" Stauchung der verklemmten Teile: $ sigma_T = " f" {self.delta_t * self.staticforce} $ " " \n "
            r" Schraubenmontagespannung: $ \sigma_M = \frac{R_{p,0.2}}{S_F \sqrt{1 + 3 (\frac{3}{d} (0.159 P + 0.577 \mu d_2)^2)}} = " f" {int(self.sigma_M)} " r" \frac{N}{mm^2} $ " " \n "
            r" maximale Torsionsspannung: $ \tau_T = \frac{M_G}{W_t} = " f" [{int(self.tauTp)}, {int(self.tauTn)}] " r" \frac{N}{mm^2} $ " " \n "
            r" maximale Zugspannung: $ \sigma_{z,max} = \frac{F_{S,max}}{A_s} = " f" [{int(self.sigma_redp)}, {int(self.sigma_redn)}] " r" \frac{N}{mm^2} $ " " \n "
            " #### Sicherheiten " " \n "
            r" Statische Sicherheit der Schraube: $ F_{S,max} = \frac{R_{p0.2}}{sigma_{z,max}} = " f" [{int(self.staticSafetyp)}, {int(self.staticSafetyn)}] $ " " \n "

        )

    def precalculation_markdown(self) -> str:
        """Wie precalculation(), liefert aber die Rechnung als Markdown."""
        (powerfactor, clampingforce, stickforce, neededAssemblyForce,
         compressionSetValue, delta_s, delta_t, pairs) = self.precalculation()
        d = self.screw.nominalDiameter
        h = self.clamping_piece.thickness
        lines = [
            "### Vorberechnung der Schraubenverbindung",
            "---",
            "#### Nachgiebigkeiten",
            rf"Nachgiebigkeit der Schraube: $ \delta_S = {_e(delta_s)} \, \frac{{\text{{mm}}}}{{\text{{N}}}} $",
            rf"Nachgiebigkeit der verspannten Teile: $ \delta_T = \frac{{h}}{{A_{{ers}} E_T}} = {_e(delta_t)} \, \frac{{\text{{mm}}}}{{\text{{N}}}} $",
            rf"Kraftverhältnis: $ \Phi_n = n \frac{{\delta_T}}{{\delta_S + \delta_T}} = {self.forceapplication} \cdot \frac{{{_e(delta_t)}}}{{{_e(delta_s)} + {_e(delta_t)}}} = {powerfactor:.3f} $",
            "#### Kräfte",
            rf"Setzbetrag: $ f_Z = 3.29 \left( \frac{{h}}{{d}} \right)^{{0.34}} \cdot 10^{{-3}} = 3.29 \left( \frac{{{h}}}{{{d}}} \right)^{{0.34}} \cdot 10^{{-3}} = {compressionSetValue:.4f} \, \text{{mm}} $",
            rf"Setzkraft: $ F_Z = \frac{{f_Z}}{{\delta_S + \delta_T}} = {_f(stickforce)} \, \text{{N}} $",
            rf"Klemmkraft: $ F_{{Kl}} = F_V - F_B (1 - \Phi_n) = {_f(clampingforce)} \, \text{{N}} $",
            rf"Erforderliche Montagekraft: $ F_{{VM}} = k_A \left( F_{{Kl}} + F_B (1 - \Phi_n) + F_Z \right) = {_f(neededAssemblyForce)} \, \text{{N}} $",
            "#### Mögliche Schrauben mit $ F_S \\geq F_{VM} $",
        ]
        if pairs:
            lines.append("\n".join(f"- {p}" for p in pairs))
        else:
            lines.append("Keine Schraube im Tabellenbereich erfüllt die Anforderung.")
        return "\n\n".join(lines)

    def markdown(self) -> str:
        """Vollständige Rechnung nach calculation() als Markdown. calculation() muss vorher aufgerufen worden sein."""
        s = self.screw
        rho_note = r"Für Ergebnisse mit zwei Angaben gilt $ [\tan(\alpha + \rho), \tan(\alpha - \rho)] $, weil dort das Gewindemoment einfliesst."
        N = r"\, \text{N}"
        Nmm = r"\, \text{Nmm}"
        MPa = r" \, \frac{\text{N}}{\text{mm}^2}"
        lines = [
            "### Schraubenverbindung",
            "---",
            f"Schraube: **{s.threadType.upper()}{s.nominalDiameter} x {s.shaftLength + s.threadLength} - {s.toughnessClass}**",
            rho_note,

            "#### Nachgiebigkeiten",
            rf"Nachgiebigkeit der Schraube: $ \delta_S = {_e(self.delta_s)} \, \frac{{\text{{mm}}}}{{\text{{N}}}} $",
            rf"Nachgiebigkeit der verspannten Teile: $ \delta_T = {_e(self.delta_t)} \, \frac{{\text{{mm}}}}{{\text{{N}}}} $",
            rf"Kraftverhältnis: $ \Phi_n = n \frac{{\delta_T}}{{\delta_S + \delta_T}} = {self.powerfactor:.3f} $",

            "#### Kräfte",
            rf"Klemmkraft: $ F_{{Kl}} = F_V - F_B (1 - \Phi_n) = {_f(self.clampingforce)} {N} $",
            rf"Setzkraft: $ F_Z = \frac{{f_Z}}{{\delta_S + \delta_T}} = \frac{{{self.compressionSetValue:.4f}}}{{{_e(self.delta_s + self.delta_t)}}} = {_f(self.stickforce)} {N} $",
            rf"Montagekraft: $ F_{{VM}} = k_A \left( F_{{Kl}} + F_B (1 - \Phi_n) + F_Z \right) = {_f(self.assemblyforce)} {N} $",
            rf"Zusatzkraft: $ F_{{SB}} = \Phi_n F_B = {_f(self.additionalForce)} {N} $",
            rf"Maximale Schraubenkraft: $ F_{{S,max}} = F_{{VM}} + F_{{SB}} = {_f(self.maxPullForce)} {N} $",
            rf"Maximal zulässige Zusatzkraft: $ F_{{S,zul}} = 0.1 \, R_{{p0.2}} \, A_S = {_f(self.allowedScrewForce)} {N} $",

            "#### Gewinde- und Anzugsmoment",
            rf"Gewindemoment: $ M_G = F_{{VM}} \frac{{d_S}}{{2}} \tan(\alpha \pm \rho) = [{_f(self.threadtorquep)}, {_f(self.threadtorquen)}] {Nmm} $",
            rf"Anzugsmoment: $ M_A = F_{{VM}} \left( \frac{{d_S}}{{2}} \tan(\alpha + \rho) + \mu_K \, d_K \right) = {_f(self.fasteningtorque)} {Nmm} $",

            "#### Spannungen",
            rf"Montagespannung: $ \sigma_M = \frac{{R_{{p0.2}} \cdot 0.9}}{{\sqrt{{1 + 3 \left( \frac{{3}}{{d}} (0.159 P + 0.577 \mu_G d_2) \right)^2}}}} = {_f(self.sigma_M)} {MPa} $",
            rf"Torsionsspannung: $ \tau_T = \frac{{M_G}}{{W_t}} = [{_f(self.tauTp, 1)}, {_f(self.tauTn, 1)}] {MPa} $",
            rf"Maximale Zugspannung: $ \sigma_{{z,max}} = \frac{{F_{{S,max}}}}{{A_S}} = {_f(self.sigma_zmax, 1)} {MPa} $",
            rf"Vergleichsspannung: $ \sigma_{{red}} = \sqrt{{\sigma_{{z,max}}^2 + 3 \tau_T^2}} = [{_f(self.sigma_redp, 1)}, {_f(self.sigma_redn, 1)}] {MPa} $",
            rf"Flächenpressung unter dem Schraubenkopf: $ p = \frac{{F_{{S,max}}}}{{A_P}} = \frac{{{_f(self.maxPullForce)}}}{{{self.screwHeadArea:.1f}}} = {_f(self.areaPressure, 1)} {MPa} $",
            rf"Dynamische Spannung: $ \sigma_a = \frac{{F_{{dyn}}}}{{A_S}} = {_f(self.dynamicStress, 1)} {MPa} $",

            "#### Sicherheiten",
            rf"Statische Sicherheit der Schraube: $ S_F = \frac{{R_{{p0.2}}}}{{\sigma_{{red}}}} = [{self.staticSafetyp:.2f}, {self.staticSafetyn:.2f}] $",
            rf"Sicherheit gegen Überlastung durch Zusatzkraft: $ S_B = \frac{{F_{{S,zul}}}}{{F_{{SB}}}} = {self.forceSafety:.2f} $",
            rf"Sicherheit gegen Flächenpressung: $ S_P = \frac{{R_e}}{{p}} = {self.areaSafety:.2f} $",
            rf"Dynamische Sicherheit: $ S_D = \frac{{\sigma_{{ASV}}}}{{\sigma_a}} = {self.dynamicSafety:.2f} $",
        ]
        return "\n\n".join(lines)
