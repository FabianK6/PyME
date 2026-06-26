#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 22:27:04 2026

@author: fabian
"""

import numpy as np
from . import bending as bd
from .materials import Solid


class SlotStone(object):
    def __init__(
        self, 
        width: float|int,
        length: float|int,
        height: float|int,
        form: str,
        material: Solid
    ):
        self.width = width
        self.length = length
        self.height = height
        self.form = form.upper()
        self.material = material
        pass
    

class Shaft(object):
    def __init__(
        self,
        length: float|int,
        diameter: float|int,
        K_A: float|int = 1.5,
        torque: float|int = 0,
        setOfNodes: list = [],
        material: Solid = None,
    ):
        """
        Shaft-Object used for stress-strain analysis

        setOfNodes should look like this:
        setOfNodes = [\n
        [0, btype, I, [Qx, Fy, Mb]], \n
        ..., \n
        [xi, btype, I, [Qxi, Fyxi, Mbxi]] \n
        ] \n
        btype can be one of the following if it is a bearing: "boundary", "mid", "rigid", "loose", "glider" \n
        and if btype is a pure load: "load" \n
        Qx is a distributed load and can contain "x" as a symbol (symoy.abc.x) [N/mm]. \n
        Fy is a point load [N]. \n
        Mb is a bending torque [Nmm].

        Args:
            length (float): length of the shaft [mm]
            diameter (float): smallest diameter of the shaft [mm]
            K_A (float, optional): Application factor. Defaults to 1.5.
            torque (float, optional): Torque acting on the shaft [Nmm]. Defaults to 0.
            setOfNodes (list, optional): bearings, forces and loads as a 2D-array. Defaults to [].
            material (Solid, optional): Material of the Shaft. Defaults to None.
        """
        self.length = length
        self.diameter = diameter
        self.K_A = K_A
        self.setOfNodes = setOfNodes
        self.torque = torque
        self.material = material
        
    def calculate_bending(
        self, 
        tol: float = 1e-26
    ):
        """
        Calculate Q(x), M(x), w(x) and W(x) where: \n
        Q(x) is the load acting on the shaft at position x \n
        M(x) is the bending moment at position x \n
        w(x) is the tilt of the axis at position x \n
        W(x) is the displacement of the axis at position x \n

        Args:
            tol (float): tolerance between x and xi. needed for calculating the position of the max bending moment.
        :returns: [[arrays Qx, Mx, wx, Wx], [callables Q(x), M(x), w(x), W(x)],

        """
        bmid = []
        for i, node in enumerate(self.setOfNodes):
            nmid = bd.SymNode(
                    subscript=i+1, 
                    position=node[0],
                    inertia=node[2],
                    young=self.material.young
                )
            match node[1]:
                case "boundary":
                    bmid.append(
                        nmid.set_as_boundarybearing().set_force(node[3][1], node[3][2]).set_load(node[3][0])
                    )
                case "mid":
                    bmid.append(
                        nmid.set_as_midbearing().set_force(node[3][1], node[3][2]).set_load(node[3][0])
                    )
                case "loose":
                    bmid.append(
                        nmid.set_as_loose().set_force(node[3][1], node[3][2]).set_load(node[3][0])
                    )
                case "rigid":
                    bmid.append(
                        nmid.set_as_rigid().set_force(node[3][1], node[3][2]).set_load(node[3][0])
                    )
                case "glider":
                    bmid.append(
                        nmid.set_as_glider().set_force(node[3][1], node[3][2]).set_load(node[3][0])
                    )
                case "load":
                        bmid.append(
                        nmid.set_force(node[3][1], node[3][2]).set_load(node[3][0])
                    )

        elements = []
        for i, b in enumerate(bmid[:-1]):
            elements.append(
                bd.SubSegment(b, bmid[i+1], i+1)
            )
        smid = bd.BeamSystem(elements, bmid)
        
        smid.solve_system()
        vmid = smid.plot_system()
        positions = np.array(vmid[2])
        forces = np.array(vmid[0][0])
        moments = np.array(vmid[0][1])
        i_xi = np.argmax(np.abs(moments))
        self.Mbmax = abs(moments[i_xi])
        self.xi = positions[i_xi]
        
        self.bearingforces = []
        for b in self.setOfNodes:
            if b[1] != "load":
                pos = b[0]
                closest = np.argmin(np.abs(positions - pos))
                if np.abs(positions[closest] - pos) < tol:
                    force = forces[closest]
                    self.bearingforces.append(force)

        return vmid, self.xi, self.bearingforces

    def set_diameter(
            self, bending: bool,
            bigBearingDistance: bool):
        """
        Berechnung des Nenndurchmessers der Welle \n
        gemäß dem Flussdiagramm aus RM 11.21

        Argumente:
            bending (bool): Ist das Biegemoment bekannt?
            bigBearingDistance (bool): Sind die Lager weit voneinander entfernt?
        """
        p = self.material.sigma_Bw / (1.73 * self.material.tau_Ds)
        if not self.torque:
            d = 3.4 * (self.Mbmax / self.material.sigma_Bw) ** (1 / 3)
        else:
            if bending:
                try:
                    self.Mv = np.sqrt((self.Mbmax) ** 2 + 0.75 * (p * self.torque * self.K_A) ** 2)
                except Exception as e:
                    if bigBearingDistance:
                        self.Mv = 2.1 * self.torque * self.K_A
                    else:
                        self.Mv = 1.17 * self.torque * self.K_A
                d = 3.4 * (self.Mv / self.material.sigma_Bw) ** (1 / 3)
            else:
                d = 2.7 * (self.torque * self.K_A / self.material.tau_Ds) ** (1 / 3)
        self.designDiameter = d

    def markdown(self):
        # tau = T / Wp -> Wp = T / tau
        Wp = self.torque * self.K_A / self.material.tau_Ds * 4
        dt = (Wp / np.pi * 16) ** (1 / 3)
        return (
            r" ### Welle " " \n "
            " *** " " \n "
            r" Position des maximalen Biegemoments: "
            f" $ x_i = {self.xi} mm $ " " \n "
            r" Maximales Biegemoment "
            r" $ M_{bmax} = M(x_i) = " f"{int(self.Mbmax)} Nmm " r" $ " " \n "
            r" Vergleichsspannung: $ M_v = " f" {int(self.Mv)} Nmm"
            r" Entwurfsdurchmesser der Welle: "
            r" $ d_{min} = " f" {round(self.designDiameter, 1)} " r" mm $ " " \n "
            r" mindestdurchmesser der Welle am Ende: "
            r" $ d_t = " f" {round(dt, 1)} mm " r" $ " " \n "
            " #### Querkräfte "  " \n "
            " *** " " \n "
            f" $ F_A, F_B, ..., F_Z = {np.round(np.array(self.bearingforces), 1)} N $ " " \n "
        )

    def calculate_polygon_connection(self, form, d1, d2, e1, c):
        """
        Berechnung der Spannung auf der Polygonwelle und der Mindestwandstärke des Außenbereichs

        Argumente:
            form (str): P3G oder P4C
            d1 (float): geometrischer Parameter laut RM/TB 12-5 [mm]
            d2 (float): geometrischer Parameter laut RM/TB 12-5 [mm]
            e1 (float): geometrischer Parameter laut RM/TB 12-5 [mm]
            c (float): Profilfaktor \n
                für P3G \n
                >>> wenn d4 <= 35 mm: c = 1,44 \n
                >>> sonst wenn d > 35 mm: c = 1,2 \n
                \n
                für P4C \n
                >>> c = 0,7

        Rückgabe:
            tuple: Druck
        """
        match form.upper():
            case "P3G":
                pressure = self.torque * self.K_A / (self.length * (0.75 * np.pi * e1 + 0.05 * d1 ** 2))
            case "P4C":
                er = (d1 - d2) / 4
                dr = d2 + 2 * er
                pressure = self.torque * self.K_A / (self.length * (np.pi * er * dr + 0.05 * dr ** 2))
        s_min = c * (self.torque * self.K_A / (1.5 * self.length * self.material.Re)) ** (1 / 2)
        return pressure, s_min
    
    def calc_slotstone_connection(
        self,
        stone: SlotStone,
        slotDept: float,
        numOfStones: int
        ):
        "TODO: see RM and RM/TB"
        roundtype = ["A", "E", "C"]
        eqTorque = self.torque * self.K_A
        self.stone = stone
        phi = 1 if numOfStones else 2
        htr = self.stone.height - slotDept
        ltr = self.stone.length - self.stone.width if self.stone.form in roundtype else self.stone.length
        q = 2 * eqTorque / (self.diameter * phi * numOfStones * htr * ltr)
        text = (
            r"  "
            )
        
class HollowShaft(Shaft):
    def __init__(
        self, length: float, 
        diameter: float,
        innerDiameter: float,
        K_A: float, 
        torque: float, 
        setOfNodes: list,
        material: Solid
    ):
        """
        Hollow-Shaft-Object used for stress-strain analysis
        
        For Nodes: use either boundary, mid, loose, rigid, glider or load as type
        you can apply forces on each type.
        setOfNodes should look like this:
        >>> setOfNodes = [
        >>>     [0, "bearing", [Qx, Fy, Mb]],
        >>>     ...,
        >>>     [xi, "load", [Qxi, Fyxi, Mbxi]]    
        >>> ]

        Args:
            length (float): length of the shaft [mm]
            diameter (float): smallest diameter of the shaft [mm]
            K_A (float, optional): Application factor. Defaults to 1.5.
            torque (float, optional): Torque acting on the shaft [Nmm]. Defaults to 0.
            setOfNodes (list, optional): bearings, forces and loads as a 2D-array. Defaults to [].
            material (Solid, optional): Material of the Shaft. Defaults to None.
        """
        
        super().__init__(
            length, diameter, 
            K_A, torque, 
            setOfNodes, material)
        
        self.innerDiameter = innerDiameter

    def set_diameter(
            self, bending: bool,
            bigBearingDistance: bool
    ):
        """
        Berechnung des Nenndurchmessers der Welle \n
        gemäß dem Flussdiagramm aus RM 11.21

        Argumente:
            bending (bool): Ist das Biegemoment bekannt?
            bigBearingDistance (bool): Sind die Lager weit voneinander entfernt?
        """
        k = self.innerDiameter / self.diameter
        p = self.material.sigma_Bw / (1.73 * self.material.tau_Ds)
        if not self.torque:
            d = 3.4 * (self.Mbmax / ((1 - k ** 4) * self.material.sigma_Bw)) ** (1 / 3)
            self.Mv = self.Mbmax * self.K_A
        else:
            if bending:
                try:
                    self.Mv = np.sqrt((self.Mbmax) ** 2 + 0.75 * (p * self.torque * self.K_A) ** 2)
                except Exception as e:
                    if bigBearingDistance:
                        self.Mv = 2.1 * self.torque * self.K_A
                    else:
                        self.Mv = 1.17 * self.torque * self.K_A
                d = 3.4 * (self.Mv / ((1 - k ** 4) * self.material.sigma_Bw)) ** (1 / 3)
            else:
                d = 2.7 * (self.torque * self.K_A / ((1 - k ** 4) * self.material.tau_Ds)) ** (1 / 3)
                self.Mv = self.torque * self.K_A
        self.designDiameter = d

    def markdown(self):
        # tau = T / Wp -> Wp = T / tau
        Wp = self.torque * self.K_A / self.material.tau_Ds * 4
        dt = (Wp / np.pi * 16 + self.innerDiameter ** 3) ** (1 / 3)
        return (
            r" ### Welle " " \n "
            " *** " " \n "
            r" Position des maximalen Biegemoments: "
            f" $ x_i = {self.xi} mm $ " " \n "
            r" Maximales Biegemoment "
            r" $ M_{bmax} = M(x_i) = " f"{int(self.Mbmax)} Nmm " r" $ " " \n "
            r" Vergleichsspannung: $ M_v = " f" {int(self.Mv)} Nmm"
            r" Entwurfsdurchmesser der Welle: "
            r" $ d_{min} = " f" {round(self.designDiameter, 1)} " r" mm $ " " \n "
            r" mindestdurchmesser der Welle am Ende: "
            r" $ d_t = " f" {round(dt, 1)} mm " r" $ " " \n "
            " #### Querkräfte "  " \n "
            " *** " " \n "
            f" $ F_A, F_B, ..., F_Z = {np.round(np.array(self.bearingforces), 1)} N $ " " \n "
        )