#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 23:13:23 2026

@author: fabian
"""

import numpy as np
from .materials import Solid

_wiretypes_ = [["SL", "SM", "DM", "SH", "DH"], ["FD", "TD", "VD"]]


class Spring(object):
    def __init__(
        self, material: Solid, 
        springRate: float
    ):
        self.material = material
        self.springRate = springRate
        pass
    

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
        Spring Object
        

        Args:
            material (Solid): material of the Spring
            Fmax (float): maximum Springforce [N]
            springRate (float): provisorical Springrate [N/mm]
            outerDiameter (float, optional): outer Diameter of the Spring [mm]. Defaults to 0 if not given.
            innerDiameter (float, optional): inner Diameter of the Spring [mm]. Defaults to 0 if not given.
            wiretype (str, optional): wiretype according to RM/TB 10-2. Defaults to "SL".
            formingtype (str, optional): "cold" for coldformed spring and "warm" for warmformed spring. Defaults to "cold".
        """
        super().__init__(material, springRate)
        self.innerDiameter = innerDiameter
        self.outerDiameter = outerDiameter
        self.wiretype = wiretype
        self.Fmax = Fmax
        self.formingtype = formingtype
        
    def calc_wireDiameter(self):
        """
        Calculate theoretical Wirediameter according to RM and choose a final wire Diameter according to RM/TB 10-2

        Returns:
            float: Wire Diameter [mm]
        """
        if self.wiretype.upper() in _wiretypes_[0]:
            k1 = 0.155
        else:
            k1 = 0.175
        assert any([self.outerDiameter > 0, self.innerDiameter > 0]), "define either outer- or inner spring-diameter."
        if self.outerDiameter > 0:
            self.wireDiameter = k1 * (self.Fmax * self.outerDiameter)**(1/3)
            self.innerDiameter = self.outerDiameter - 2 * self.wireDiameter
        elif self.innerDiameter > 0:
            k2 = 2 * (k1 * (self.Fmax * self.innerDiameter)**(1/3))**2 / (3 * self.innerDiameter)
            self.wireDiameter = k1 * (self.Fmax * self.innerDiameter)**(1/3) + k2
            self.outerDiameter = self.innerDiameter + 2 * self.wireDiameter
        self.nominalDiameter = (self.outerDiameter + self.innerDiameter) / 2
        assert self.nominalDiameter > 0, "Given Inner Diameter too small. Spring cannot sustain Fmax."
        return self.wireDiameter
    
    def calc_spring(self, truewireDiameter: float, maxWireDiameter: float, k: float):
        """
        Calculate Spring parameters according to RM

        Args:
            truewireDiameter (float): chosen wire Diameter [mm]
            maxWireDiameter (float): maximum wire Diameter d + es according to RM/TB 10-2 [mm]
            k (float): dynamic stress factor according to RM/TB 10-15
        """
        self.truewireDiameter = truewireDiameter
        self.maxWireDiameter = maxWireDiameter
        n_ = self.material.shear * self.truewireDiameter**4 / (8 * self.nominalDiameter**3 * self.springRate)
        self.windings = round(n_)
        self.trueSpringRate = self.material.shear * self.truewireDiameter**4 / (8 * self.windings * self.nominalDiameter**3)
        self.trueSpringPitch = 8 * self.nominalDiameter**3 * self.windings * self.Fmax / (self.material.shear * self.truewireDiameter**4)

        if self.formingtype.lower() == "warm":
            self.trueWindings = self.windings + 1.5
            self.blockPitch = self.trueWindings * self.maxWireDiameter
            self.Sa = 0.02 * (self.nominalDiameter + self.truewireDiameter) * self.windings
            self.Sa_dyn = 2 * self.Sa
        elif self.formingtype.lower() == "cold":
            self.trueWindings = self.windings + 2
            self.blockPitch = (self.trueWindings - 0.3) * self.maxWireDiameter
            a = (self.nominalDiameter**2 / self.truewireDiameter)
            self.Sa = (0.0015 * a + 0.1 * self.truewireDiameter) * self.windings
            self.Sa_dyn = 1.5 * self.Sa
            
        self.maxPitch = self.blockPitch + self.Sa
        self.maxPitch_dyn = self.blockPitch + self.Sa_dyn
        
        W = np.pi * self.truewireDiameter**3 / 16
        F1 = self.trueSpringRate * 0.1 * self.trueSpringPitch
        self.tau_12 = np.array([F1, self.Fmax]) * self.nominalDiameter/(2 * W)
        self.tau_k12 = k * self.tau_12
        self.tau_kh = abs(np.diff(self.tau_k12))