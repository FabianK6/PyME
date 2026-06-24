#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 23:15:51 2026

@author: fabian
"""

import numpy as np
from .materials import Solid


class Pin(object):
    def __init__(self,
        diameter: float,
        pinLength: float,
        insertionDept: float,
        material: Solid
    ):
        """
        Stahlpin für Verbindungen oder Positionierungen von Bauteilen.

        Parameters:
            diameter (float): Pindurchmesser d [mm]
            pinLength (float): Pinlänge l [mm]
            insertionDept (float): Einstecktiefe im Bauteil mit Pressverbindung s [mm]
            material (Solid): Material des Pins
        """
        self.diameter = diameter
        self.pinLength = pinLength
        self.insertionDept = insertionDept
        self.material = material
        self.resistance = self.diameter**3 * np.pi / 16
        self.crossSection = self.diameter**2 * np.pi / 4
        self.maxShearForce = 0.6 * self.crossSection * self.material.Rm / 1.25
    
    def rough_dimensioning(
        self,
        clampingFactor,
        allowedStress,
        force,
        K_A = 1.5
    ):
        """
        Überschlagsmässige Durchmesserbestimmung eines Bolzens je nach Anwendungsfall
        Roloff Matek: Gleichung 9.1

        für clampingfactor gilt:
        1;6(1,9) für Einbaufall 1 (Bolzen lose in Stange und Gabel)
        1;1(1,4) für Einbaufall 2 (Bolzen mit Übermaßpassung in der Gabel)
        1;1(1,2) für Einbaufall 3 (Bolzen mit Übermaßpassung in der Stange)
        
        Args:
            clampingFactor (float): berücksichtigung der Einbauart gemäss Beschreibung
            allowedStress (float): Zulässige Spannung (für nicht gehärtete Bolzen: 400 MPa als Richtwert)
            force (float): zu erwartende Schärkraft
            K_A (float): Anwendungsfaktor nach RM TB 3-4
        """
        term = K_A * force / allowedStress
        self.diameter = clampingFactor * np.sqrt(term)
        

class PinConnection(object):
    def __init__(
        self,
        pin: Pin,
        part1: list,
        part2: list,
        num_of_pins: int = 1,
    ):
        """
        Berechnung einer Pinverbindung zwischen zwei Werkstücken.
        
        Anwendungen:
            1: Drehmomentübertragung mit exzentrisch angeordneten Pins an einem Flansch
            2: Positionierung zweier Werkstücke wobei auf eines eine seitliche Kraft wirkt
        
        Args:
            pin (Pin): Pin-Objekt
            num_of_pins (int): Anzahl Pins in der Verbindung n
            part1 (list): erstes Werkstück [Dicke t_1, Re_1]
            part2 (list): zweites Werkstück [Dicke t_2, Re_2]
        """
        self.pin = pin
        self.num_of_pins = num_of_pins
        self.part1 = part1
        self.part2 = part2

    def calc_torque_transition(
            self, bore_set_diameter: float|int,
            K_A: float|int, torque: float|int):
        """

        Parameters:
            bore_set_diameter (float|int): Lochkreisdurchmesser D [mm]
            K_A (float): Anwendungsfaktor nach RM TB 3-4
            torque (float): Zu übertragendes Drehmoment T [Nmm]

        Returns:
            (tuple): ([sigma_b, tau_a, p_1, p_2], [SF_b, SF_a, SF_1, SF_2], text)

        """
        torque_eq = torque * K_A
        force = torque_eq * bore_set_diameter / (2 * self.num_of_pins)

        bend = force * (self.pin.pinLength - self.pin.insertionDept) / self.pin.resistance
        shear = force / self.pin.crossSection
        contact_pressure1 = force / (self.pin.diameter * self.part1[0])
        contact_pressure2 = force / (self.pin.diameter * self.part2[0])

        SF_b = self.pin.material.Re / bend
        SF_a = self.pin.material.Re / shear
        SF_1 = self.part1[1] / contact_pressure1
        SF_2 = self.part2[1] / contact_pressure2

        text = (
            r" ### Ergebnisse: Pinverbindung " " \n " " *** " " \n "
            r" Pin: " f" {self.pin.diameter} x {self.pin.pinLength} - {self.pin.insertionDept} " " \n "
            " #### Aktionen " " \n " " *** " " \n "
            r" Equivalentes Drehmoment: $ T_{eq} = K_A T = " f" {torque_eq} Nmm $ " " \n "
            r" Querkraft pro Pin: $ F_Q = \frac{d T_{eq}}{2 n} = " f" {force} N $ " " \n "
            r" #### Reaktionen " " \n " " *** " " \n " 
            r" Durch die Bewegungsfreiheit in der Bohrung im Gegenstück kann ein Biegemoment auf den Pin wirken " " \n "
            r" Biegespannung: $ \sigma_b = \frac{16 F_Q (l - s)}{d³ \pi} = " f" {bend} Nmm $ " " \n "
            r" Schärspannung im Pin: $ \tau_a = \frac{4 F_Q}{d² \pi} = " f" {shear} " r" \frac{N}{mm²} $ " " \n "
            r" Lochleibung im Bauteil 1: $ p_1 = \frac{F_Q}{d t_1} = " f" {contact_pressure1} " r" \frac{N}{mm²} $ " " \n "
            r" Lochleibung im Bauteil 2: $ p_2 = \frac{F_Q}{d t_2} = " f" {contact_pressure2} " r" \frac{N}{mm²} $ " " \n "
            r" #### Sicherheiten " " \n " " *** " " \n "
            r" Sicherheit gegen Fliessen bei Biegung: $ S_F = \{Re}{\sigma_b} = " f" {SF_b} $ " " \n "
            r" Sicherheit gegen Fliessen bei Schärung: $ S_F = \{Re}{\tau_a} = " f" {SF_a} $ " " \n "
            r" Sicherheit gegen Fliessen bei Lochleibung im ersten Teil: $ S_F = \{Re_1}{p_1} = " f" {SF_1} $ " " \n "
            r" Sicherheit gegen Fliessen bei Lochleibung im zweiten Teil: $ S_F = \{Re_2}{p_2} = " f" {SF_2} $ " " \n "
        )
        return [bend, shear, contact_pressure1, contact_pressure2], [SF_b, SF_a, SF_1, SF_2], text
