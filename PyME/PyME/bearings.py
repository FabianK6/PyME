#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 23:07:52 2026

@author: fabian
"""

import numpy as np

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
            frequency (float|np.ndarray): [Hz]
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
        Parr = (self.radialForce * self.radialFactor + self.axialForce * self.radialFactor)**self.p
        self.load = np.sum(Parr * self.frequency / nm * percentages)**(1/self.p)
        fn = ((33+1/3)/nm)**(1/self.p)
        self.carryNumber = self.load * fL / fn
        self.L10 = (self.carryNumber/self.load)**self.p
        self.L10h = 1e6 * self.L10 / (60 * nm)
        
    
class SlideBearing(Bearing):
    def __init__(self, 
        innerDiameter: float,
        outerDiameter: float,
        axialForce: float,
        radialForce: float,
        frequency: float,
        widthRatio: float,
        kinematicViscosity: float,
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
            relativeExcentricity (float): [mm]
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
        self.kinematicViscosity = kinematicViscosity
        self.relativeExcentricity = relativeExcentricity
        self.beta = beta
        
    def calc_with_constant_load(self):
        self.width = self.widthRatio * self.innerDiameter
        self.pL = self.radialForce / (self.innerDiameter * self.width)
        
        self.uW = self.innerDiameter * np.pi * self.frequency
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
        
        d = (self.kinematicViscosity * self.omega)
        self.sommerfeld = self.pL * self.psi_B**2 / d
        
        m = np.pi / (self.sommerfeld * np.sqrt(1 - self.relativeExcentricity**2))
        n = self.relativeExcentricity/2 * np.sin(self.beta)
        self.friction = self.psi_B * (m + n)
        
    def flowrate(self, Vp_rel):
        a = self.widthRatio - 0.223 * self.widthRatio**3
        Vdrel = self.relativeExcentricity * 0.25 * a
        self.VD = Vdrel * self.innerDiameter**3 * self.psi_B * self.omega
        self.Vp = Vp_rel * self.innerDiameter**3 * self.psi_B**3 / self.kinematicViscosity
        return self.Vp + self.VD
    
    def temperature(self, flowrate, T_in, spezHeatCap, density):
        self.PR = self.mu * self.radialForce * self.uW
        return T_in + self.PR / (spezHeatCap * flowrate * density)
  