#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 23:02:37 2026

@author: fabian
"""

import numpy as np

_QUALITYFACTOR_STRAIGHT_ = {
    6: {1: 9.6, 2: 0.0193},
    7: {1: 15.3, 2: 0.0193},
    8: {1: 24.5, 2: 0.0193},
    9: {1: 32.5, 2: 0.0193},
    10: {1: 53.6, 2: 0.0193},
    11: {1: 76.6, 2: 0.0193},
    12: {1: 122.5, 2: 0.0193}
}

_QUALITYFACTOR_TILTED_ = {
    6: {1: 8.5, 2: 0.0087},
    7: {1: 13.6, 2: 0.0087},
    8: {1: 21.8, 2: 0.0087},
    9: {1: 30.7, 2: 0.0087},
    10: {1: 47.7, 2: 0.0087},
    11: {1: 68.2, 2: 0.0087},
    12: {1: 109.1, 2: 0.0087}
}

_FOREHEADFACTORS_ = {
    6: {1: 1.0, 2: 1.0},
    7: {1: 1.0, 2: 1.1},
    8: {1: 1.1, 2: 1.2},
    9: {1: 1.2, 2: 1.4},
    10: {1: 1.22, 2: 1.42},
    11: {1: 1.24, 2: 1.44},
    12: {1: 1.26, 2: 1.46}
}

class Gear(object):
    def __init__(
        self, 
        teeth: float, 
        module: float, 
        width: float, 
        toothWidth: float,
        distanceGearToshaftMidPoint: float, 
        toothingQuality: int,
        Kdot: float, 
        shaft=None,
        alpha = 20
    ):
        """
        Gear object.
        Input Units in mm, degree

        Args:
            teeth (int): z
            module (float|int): m
            width (float|int): b
            shaft (Shaft): is an object. Defaults to None
            distanceGearToshaftMidPoint (float|int): l
            alpha (int, optional): a. Defaults to 20.
        
        Attributes:
            partCircleDiameter: d
            baseCircleDiameter: db
            headCircleDiameter: da
            feetCircleDiameter: df
            teethFeetHeight: hf
            teethHeight: h
        """
        self.teeth = teeth
        self.alpha = alpha / 180 * np.pi
        self.module = module
        self.toothWidth = toothWidth
        self.width = width
        self.distanceGearToshaftMidPoint = distanceGearToshaftMidPoint
        self.toothingQuality = toothingQuality
        self.Kdot = Kdot
        self.shaft = shaft
        self.qH = {
            6: 1.32, 7: 1.85, 8: 2.59, 
            9: 4.01, 10: 6.22, 11: 9.63, 
            12: 14.9
        }.get(toothingQuality)
    
    def set_nominalCircumferalForce(self, torque: float):
        self.nominalCircumferalForce = 2 * torque / self.partCircleDiameter
        return self.nominalCircumferalForce
        
    def set_radialForce(self):
        """
        Radialkraft auf das Zahnrad.
        Berechne die Umfangskraft zuerst.

        Returns:
            _type_: _description_
        """
        self.radialForce = self.nominalCircumferalForce * np.tan(self.alpha) / np.cos(self.beta)
        return self.radialForce
        
    def set_axialForce(self):
        """
        Axialkraft auf das Zahnrad.
        berechne die Radialkraft zuerst.

        Returns:
            _type_: _description_
        """
        self.axialForce = self.radialForce * np.tan(self.beta)
        return self.axialForce
        
    def set_lineForce(self, K_A: float):
        """
        Linienkraft auf ein Zahn.
        Berechne die Umfangskraft zuerst.

        Args:
            K_A (float): application factor

        Returns:
            _type_: _description_
        """
        self.lineForce = K_A * (self.nominalCircumferalForce / self.toothWidth)
        return self.lineForce
            
    def set_flankDeviation(self):
        a = self.shaft.length * self.distanceGearToshaftMidPoint/ self.baseCircleDiameter
        b = (self.baseCircleDiameter / self.shaft.diameter)**4
        c = (self.width / self.baseCircleDiameter)**2
        self.flankDeviation = 0.023 * self.lineForce * (0.3 + abs(0.7 + self.Kdot * a * b)) * c
        return self.flankDeviation
    
    def set_dynamicFactor(self, translation: float):
        K3 = 0.01 * self.teeth * self.headCircleDiameter * np.pi * self.speed * np.sqrt(translation**2 / (1 + translation**2))
        self.Kv = 1 + K3 * (self.K1 / self.lineForce + self.K2)
        return self.Kv
    
    def set_speed(self, speed: float):
        self.speed = speed
        return speed
    
    def set_flanklineDeviation(self):
        a = (self.partCircleDiameter / self.shaft.diameter)**4
        b = abs(0.7 + self.Kdot * (self.shaft.length * self.distanceGearToshaftMidPoint / self.partCircleDiameter**2) * a)
        self.flanklineDeviation = 0.023 * self.lineForce * (b + 0.3) * (self.width / self.partCircleDiameter)**2
        return self.flanklineDeviation
    
    def set_K_Fges(self, K_A: float):
        self.K_Fges = K_A * self.Kv * self.K_Fa * self.K_Ha
        return self.K_Fges
    
    def set_K_Hges(self, K_A: float):
        self.K_Hges = np.sqrt(K_A * self.Kv * self.K_Fa * self.K_Ha)
        return self.K_Hges
    
    def set_sigma_F(self, Y_Fa: float, Y_Sa: float, epsilon_alpha: float, epsilon_beta: float):
        Y_epsilon = 0.25 + 0.75/(epsilon_alpha / np.cos(self.beta)**2)
        Y_beta = 1 - epsilon_beta * self.beta / (np.pi*2/3)
        self.sigma_F = self.nominalCircumferalForce / (self.toothWidth * self.module) * Y_Fa * Y_Sa * Y_epsilon * Y_beta * self.K_Fges
        return self.sigma_F


class StraightGear(Gear):
    def __init__(
        self,
        teeth: float,
        module: float,
        width: float,
        toothWidth: float,
        distanceGearToshaftMidPoint: float,
        toothingQuality: int,
        Kdot: float,
        shaft=None,
        alpha=20,
    ):
        super().__init__(teeth, module, width, toothWidth,
                         distanceGearToshaftMidPoint, toothingQuality, Kdot, shaft, alpha)
        self.K1 = qualityfactor.get(toothingQuality).get(1)
        self.K2 = qualityfactor.get(toothingQuality).get(2)
        self.divisionOfIntervention = np.pi * module * np.cos(self.alpha)
        self.partCircleDiameter = teeth * module / np.cos(self.beta)
        self.baseCircleDiameter = teeth * module * np.cos(self.alpha) / np.cos(self.beta)
        self.headCircleDiameter = module * (teeth + 2)
        self.feetCircleDiameter = module * (teeth - 2.5)
        self.teethFeetHeight = module * 1.25
        self.teethHeight = module * 2.25
        self.jump = self.width * np.tan(self.beta)
        self.jumpOverlap = self.width * np.sin(self.beta) / (np.pi * self.module)
        self.K_Fa = foreheadfactors.get(self.toothingQuality).get(1)
        self.K_Ha = foreheadfactors.get(self.toothingQuality).get(1)

class TiltedGear(Gear):
    def __init__(
            self,
            teeth: float,
            module: float,
            width: float,
            toothWidth: float,
            distanceGearToshaftMidPoint: float,
            toothingQuality: int,
            Kdot: float,
            beta: int | float,
            shaft=None,
            alpha=20
    ):
        super().__init__(teeth, module, width, toothWidth,
                         distanceGearToshaftMidPoint, toothingQuality, Kdot, shaft, alpha)
        self.beta = beta / 180 * np.pi
        self.K1 = qualityfactor.get(toothingQuality).get(1)
        self.K2 = qualityfactor.get(toothingQuality).get(2)
        self.frontModule = self.module / np.cos(self.beta)
        self.frontalpha = np.atan(np.tan(self.alpha) / np.cos(self.beta))
        self.baseCircleDiameter = self.partCircleDiameter * np.cos(self.frontalpha)
        self.headCircleDiameter = self.module * (2 + teeth / np.cos(self.alpha))
        self.feetCircleDiameter = self.partCircleDiameter - 2.5 * module
        self.divisionOfIntervention = np.pi * self.frontModule * np.cos(self.alpha)
        self.K_Fa = foreheadfactors.get(self.toothingQuality).get(2)
        self.K_Ha = foreheadfactors.get(self.toothingQuality).get(2)
    
    
class Gearbox(object):
    def __init__(
        self, gear1: Gear, gear2: Gear,
        torque_IN: float|int, K_A, speed,
        Y_Fa1, Y_Fa2, Y_Sa1, Y_Sa2
    ):
        """
        design a Gearbox

        Args:
            gear1 (Gear): Driving Gear
            gear2 (Gear): To be driven Gear
            torque_IN (float | int): Torque going into the driver in [Nm]
            
        Attributes:
            translation: i
            axisDistance: ad
            intersection: ga
            profileOverlap: ea
        """
        self.gear1 = gear1
        self.gear2 = gear2
        self.T_eq = torque_IN * 1000 * K_A
        self.K_A = K_A
        self.Y_Fa1 = Y_Fa1
        self.Y_Fa2 = Y_Fa2
        self.Y_Sa1 = Y_Sa1
        self.Y_Sa2 = Y_Sa2
        # Geometrieparameter
        self.translation = self.gear2.teeth / self.gear1.teeth
        gear1.set_speed(speed)
        gear2.set_speed(speed / self.translation)
        self.axisDistance = (self.gear1.partCircleDiameter + self.gear2.partCircleDiameter) / 2
        a = (self.gear1.headCircleDiameter**2 - self.gear1.baseCircleDiameter**2)**(1/2)
        b = self.gear2.teeth / abs(self.gear2.teeth) * (self.gear2.headCircleDiameter**2 - self.gear2.baseCircleDiameter**2)**(1/2)
        if gear1.beta == 0:
            c = self.axisDistance * np.sin(self.gear1.alpha)
        else:
            c = self.axisDistance * np.sin(self.gear1.frontalpha)
        self.intersection = 1/2 * (a + b) - c
        self.totalOverlap = self.intersection + self.gear1.jumpOverlap
        self.profileOverlap = self.intersection / self.gear1.divisionOfIntervention
        self.stepOverlap = self.gear1.toothWidth * np.sin(self.gear1.beta) / (np.pi * self.gear1.module)
        # Kräfte
        self.gear1.set_nominalCircumferalForce(self.T_eq)
        self.gear2.set_nominalCircumferalForce(self.T_eq * self.translation)
        self.gear1.set_radialForce()
        self.gear2.set_radialForce()
        self.gear1.set_axialForce()
        self.gear2.set_axialForce()
        self.gear1.set_lineForce(self.applicationFactor)
        if self.gear1.lineForce <= 100:
            self.gear1.set_lineForce(100)
        self.gear2.set_lineForce(self.applicationFactor)
        if self.gear2.lineForce <= 100:
            self.gear2.set_lineForce(100)
        self.gear1.set_flankDeviation()
        # Festigkeitsnachweis
        self.gear1.set_dynamicFactor(self.translation)
        self.gear2.set_dynamicFactor(self.translation)
        self.gear1.set_K_Fges(K_A=self.K_A)
        self.gear2.set_K_Fges(K_A=self.K_A)
        self.gear1.set_sigma_F(Y_Fa=self.Y_Fa1, Y_Sa=Y_Sa1, epsilon_alpha=self.profileOverlap, epsilon_beta=self.stepOverlap)
        self.gear2.set_sigma_F(Y_Fa=self.Y_Fa2, Y_Sa=self.Y_Sa2, epsilon_alpha=self.profileOverlap, epsilon_beta=self.stepOverlap)

        
def calcModule(
    case: str, shaftDiameter = None, 
    beta = None, teeth = None, 
    axisDistance = None, translation = None, 
    torque = None, diameterWidthRatio = None,
    feetStrength = None, flankStrength = None,
    applicationFactor = None
):
    """
    calc module from a given case\n
    
    possible cases:\n
    >>> "gear-on-shaft"
    >>> "geared-shaft"
    >>> "axis-distance-given"
    >>> "flank-hardened"
    >>> "flank-soft"
    
    arguments used for case "gear-on-shaft" or "geared-shaft":\n
    >>> shaftDiameter, beta, teeth
    
    arguments used for case "axis-distance-given":
    >>> axisDistance, beta, translation, teeth
    
    arguments used for case "flank-hardened":
    >>> torque, applicationFactor, beta, teeth, diameter_with_ratio, feet_strength
    
    arguments used for case "flank-soft":
    >>> torque, applicationFactor, beta, teeth, diameter_with_ratio, flank_strength
    """
    match case:
        case "gear-on-shaft":
            m = 1.8 * shaftDiameter * np.cos(beta) / (teeth - 2.5)
            return m
        case "geared-shaft":
            m = 1.1 * shaftDiameter * np.cos(beta) / (teeth - 2.5)
        case "axis-distance-given":
            m = 2 * axisDistance * np.cos(beta) / (1 + translation) / teeth
        case "flank-hardened":
            a = torque * applicationFactor * np.cos(beta)**2
            b = teeth**2 * diameterWidthRatio * feetStrength
            m = 1.85 * (a / b)**(1/3)
        case "flank-soft":
            a = 98 * np.cos(beta) / teeth
            b = torque * applicationFactor / (diameterWidthRatio * flankStrength**2)
            c = (translation + 1) / translation
            m = a * (b * c)**(1/3)
    return m