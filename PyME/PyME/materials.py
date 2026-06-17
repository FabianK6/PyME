#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 22:51:36 2026

@author: fabian
"""

import numpy as np

class Fluid(object):
    def __init__(self, density20, kinematicViscosity20, spezHeatCap, heatexpansion, baseTemp = 20):
        self.density20 = density20
        self.kinematicViscosity20 = kinematicViscosity20
        self.spezHeatCap = spezHeatCap
        self.heatexpansion = heatexpansion
        self.baseTemp = 273.15 + baseTemp
        
    def density_p(self, pressure, bulk_modulus):
        return self.density20 / (1 - pressure / bulk_modulus)
    
    def density_T(self, temperature):
        return self.density20 / (1 + self.heatexpansion * (temperature - 20))
    
    def density_pT(self, pressure, temperature, bulk_modulus):
        return self.density_T(temperature) * (1 + 1/bulk_modulus * pressure)
    
    
class SRSCalibrationFluid(Fluid):
    def __init__(self):
        super().__init__(814, 4e-6, 1950, 8.22e-4)
        
    def viscosity_T(self, temperature):
        a = (2.5e-6 - 4e-6)/20
        dT = temperature - self.baseTemp
        return self.kinematicViscosity20 + dT * a
        
    def bulk_modulus(self, pressure):
        return (8427.792 + 18528.471 * np.log(1 + 0.000642 * pressure * 1e-5)) * 1e5
    
    def speedOfSound(self, pressure):
        K = self.bulk_modulus(pressure)
        density = self.density_p(pressure, K)
        return np.sqrt(K/density)
    
    
class ShellRimulaVG30(Fluid):
    def __init__(self):
        super().__init__(876, 116e-4, 2050, 0.65e-3)
    
    def viscosity_T(self, temperature):
        a = (116e-4 - 15.3e-4)/60
        dT = temperature - self.baseTemp
        return self.kinematicViscosity20 + dT * a


class Water(Fluid):
    def __init__(self):
        super().__init__(998, 1e-6, 4185, 0.2568e-3)
        
    def viscosity_T(self, temperature):
        a = (1e-6 - 0.4415e-6)/40
        dT = temperature - self.baseTemp
        return self.kinematicViscosity20 + dT * a
    
    def bulk_modulus(pressure):
        return 2.2e9
    
    
class Solid(object):
    def __init__(
            self, density: float, 
            young: float, shear: float, poisson: float, 
            Rm: float, Re: float, 
            sigma_ZDw: float,sigma_ZDs: float,
            sigma_Bw: float, sigma_Bs: float,
            tau_Dw: float, tau_Ds: float):
        """
        Solid Material.

        Parameters
        ----------
        density : float
            Dichte des Werkstoffs [kg/m³].
        young : float
            Elastizitätsmodul [N/mm²].
        shear : float
            Querdehnmodul [N/mm²].
        poisson : float
            Querkontraktionszahl.
        Rm : float
            Zugfestigkeit [N/mm²].
        Re : float
            Streckgrenze [N/mm²].
        sigma_ZDw : float
            Dynamische Zug/Druck Wechselfestigkeit [N/mm²].
        sigma_ZDs : float
            Dynamische Zug/Druck Schwellfestigkeit [N/mm²].
        sigma_Bw : float
            Dynamische Biege-Wechselfestigkeit [N/mm²].
        sigma_Bs : float
            Dynamische Biege-Schwellfestigkeit [N/mm²].
        tau_Dw : float
            Dynamische Torsions-Wechselfestigkeit [N/mm²].
        tau_Ds : float
            Dynamische Torsions-Schwellfestigkeit [N/mm²].

        Returns
        -------
        None.

        """
        self.density = density
        self.young = young
        self.shear = shear
        self.poisson = poisson
        self.Rm = Rm
        self.Re = Re
        self.sigma_ZDw = sigma_ZDw
        self.sigma_ZDs = sigma_ZDs
        self.sigma_Bw = sigma_Bw
        self.sigma_Bs = sigma_Bs
        self.tau_Dw = tau_Dw
        self.tau_Ds = tau_Ds
        
    def dynamic(self, mode, d1, d2, tk, SD):
        match mode:
            case "ZD2":
                sigma = self.tau_Ds
            case "ZD3":
                sigma = self.tau_Dw
            case "t2":
                sigma = self.sigma_Ds
            case "t3":
                sigma = self.sigma_Dw
            case "B2":
                sigma = self.sigma_Bs
            case "B3":
                sigma = self.sigma_Bw
        return d1 * d2 * sigma / (tk, SD)


class Steel_S235(Solid):
    def __init__(
            self, density=7800, young=210000, shear=80000,
            poisson=0.3, Re=235, Rm=360,
            sigma_ZDw=140, sigma_ZDs=235,
            sigma_Bw=180, sigma_Bs=280,
            tau_Dw=105, tau_Ds=165):
        """
        Standardwerkstoff im Maschinen- und Stahlbau, bei
        mäßiger Beanspruchung; Flach- und Langerzeugnisse; gut bearbeitbar, Schweißeignung verbessert sich
        bei jeder Sorte von Gütegruppe JR bis K

        Parameters
        ----------
        density : TYPE, optional
            DESCRIPTION. The default is 7800.
        young : TYPE, optional
            DESCRIPTION. The default is 210000.
        poisson : TYPE, optional
            DESCRIPTION. The default is 0.3.
        Re : TYPE, optional
            DESCRIPTION. The default is 235.
        Rm : TYPE, optional
            DESCRIPTION. The default is 360.
        sigma_ZDw : TYPE, optional
            DESCRIPTION. The default is 140.
        sigma_ZDs : TYPE, optional
            DESCRIPTION. The default is 235.
        sigma_Bw : TYPE, optional
            DESCRIPTION. The default is 180.
        sigma_Bs : TYPE, optional
            DESCRIPTION. The default is 280.
        tau_Dw : TYPE, optional
            DESCRIPTION. The default is 105.
        tau_Ds : TYPE, optional
            DESCRIPTION. The default is 165.

        Returns
        -------
        None.

        """

        super().__init__(
            density, young, shear, poisson, 
            Rm, Re, 
            sigma_ZDw, sigma_ZDs, 
            sigma_Bw, sigma_Bs, 
            tau_Dw, tau_Ds)
        
class Steel_30CrNiMo8(Solid):
    def __init__(
        self, density=7800, young=210000, shear=80000, 
        poisson= 0.3, Rm=1250, Re=1050, 
        sigma_ZDw=500, sigma_ZDs=750, 
        sigma_Bw=625, sigma_Bs=935, 
        tau_Dw=375, tau_Ds=625
    ):
        """
        höchstbeanspruchte Bauteile im Fahrzeug- und \n
        Maschinenbau; große Getriebewellen, Turbinenläufer, Zahnräder
        """
        super().__init__(
            density, young, shear, poisson, 
            Rm, Re, sigma_ZDw, sigma_ZDs, 
            sigma_Bw, sigma_Bs, 
            tau_Dw, tau_Ds)
        

class Steel_C15E(Solid):
    def __init__(
            self, density=7800, young=210000, shear=80000, 
            poisson=0.3, Rm=800, Re=545, 
            sigma_ZDw=320, sigma_ZDs=540, 
            sigma_Bw=400, sigma_Bs=655, 
            tau_Dw=240, tau_Ds=380):
        """
        Direkt härtbare kleine Teile mit niedriger Kernfestigkeit; \n
        Bolzen, Buchsen, Zapfen, Hebel, Gelenke, Spindeln
        """
        super().__init__(
            density, young, shear, poisson, 
            Rm, Re, 
            sigma_ZDw, sigma_ZDs, 
            sigma_Bw, sigma_Bs, 
            tau_Dw, tau_Ds)
        
        
class Steel_42CrMo4(Solid):
    def __init__(
            self, density=7800, young=210000, shear=80000,
            poisson=0.3, Re=900, Rm=1100, 
            sigma_ZDs=440, sigma_ZDw=685, 
            sigma_Bw=550, sigma_Bs=855,
            tau_Dw=330, tau_Ds=565):
        """
        größere Querschnitte des Maschinen- und Fahrzeugbaus 
        mit hoher Kernfestigkeit; Getriebewellen, Keilwellen

        Parameters
        ----------
        density : TYPE, optional
            DESCRIPTION. The default is 7800.
        young : TYPE, optional
            DESCRIPTION. The default is 210000.
        poisson : TYPE, optional
            DESCRIPTION. The default is 0.3.
        Re : TYPE, optional
            DESCRIPTION. The default is 900.
        Rm : TYPE, optional
            DESCRIPTION. The default is 1100.
        sigma_ZDs : TYPE, optional
            DESCRIPTION. The default is 440.
        sigma_ZDw : TYPE, optional
            DESCRIPTION. The default is 685.
        sigma_Bw : TYPE, optional
            DESCRIPTION. The default is 550.
        sigma_Bs : TYPE, optional
            DESCRIPTION. The default is 855.
        tau_Dw : TYPE, optional
            DESCRIPTION. The default is 330.
        tau_Ds : TYPE, optional
            DESCRIPTION. The default is 565.

        Returns
        -------
        None.

        """
        super().__init__(
            density, young, shear, poisson, 
            Rm, Re, 
            sigma_ZDw, sigma_ZDs, 
            sigma_Bw, sigma_Bs, 
            tau_Dw, tau_Ds)
        
        
class Steel_X5CrNi18_10(Solid):
    def __init__(
            self, density=7800, young=210000, shear=80000, 
            poisson=0.3, Rm=500, Re=190, 
            sigma_ZDw=200, sigma_ZDs=200, 
            sigma_Bw=250, sigma_Bs=250, 
            tau_Dw=150, tau_Ds=150):
        """
        Universeller Einsatz;
        Bauwesen, Fahrzeugbau,
        Lebensmittelindustrie

        Parameters
        ----------
        density : TYPE, optional
            DESCRIPTION. The default is 7800.
        young : TYPE, optional
            DESCRIPTION. The default is 210000.
        poisson : TYPE, optional
            DESCRIPTION. The default is 0.3.
        Rm : TYPE, optional
            DESCRIPTION. The default is 500.
        Re : TYPE, optional
            DESCRIPTION. The default is 190.
        sigma_ZDw : TYPE, optional
            DESCRIPTION. The default is 200.
        sigma_ZDs : TYPE, optional
            DESCRIPTION. The default is 200.
        sigma_Bw : TYPE, optional
            DESCRIPTION. The default is 250.
        sigma_Bs : TYPE, optional
            DESCRIPTION. The default is 250.
        tau_Dw : TYPE, optional
            DESCRIPTION. The default is 150.
        tau_Ds : TYPE, optional
            DESCRIPTION. The default is 150.

        Returns
        -------
        None.

        """
        super().__init__(
            density, young, shear, poisson, 
            Rm, Re, 
            sigma_ZDw, sigma_ZDs, 
            sigma_Bw, sigma_Bs, 
            tau_Dw, tau_Ds)
        
class Alu_ENAW6060(Solid):
    def __init__(
            self, density=2700, young=72000, shear=24000, 
            poisson=0.1, Re=150, Rm=195,
            sigma_ZDw=120, sigma_ZDs=150, 
            sigma_Bw=105, sigma_Bs=135,
            tau_Dw=65, tau_Ds=85):
        """
        Die Sorten der Reihe 6000 sind kalt und warm aushärtbar,
        schweißbar, korrosionsbeständig, nicht dekorativ anodisierbar, 
        die Sorte 6060 ist darüber hinaus besonders gut
        strangpressbar, auch ist ein Aushärten nach dem Schweißen möglich; 
        Profile für Tragkonstruktionen, Fenster-,
        Tür-, Abdeck- und Abschlussprofile, Rollladenstäbe,
        Heizkörper, Maschinentische, Elektromotorengehäuse,
        Pneumatikzylinder, Aufbauten, Container, Einrichtungen
        von Schiffen und Schienenfahrzeugen

        Parameters
        ----------
        density : TYPE, optional
            DESCRIPTION. The default is 2700.
        young : TYPE, optional
            DESCRIPTION. The default is 72000.
        poisson : TYPE, optional
            DESCRIPTION. The default is 0.1.
        Re : TYPE, optional
            DESCRIPTION. The default is 150.
        Rm : TYPE, optional
            DESCRIPTION. The default is 195.
        sigma_ZDw : TYPE, optional
            DESCRIPTION. The default is 120.
        sigma_ZDs : TYPE, optional
            DESCRIPTION. The default is 150.
        sigma_Bw : TYPE, optional
            DESCRIPTION. The default is 105.
        sigma_Bs : TYPE, optional
            DESCRIPTION. The default is 135.
        tau_Dw : TYPE, optional
            DESCRIPTION. The default is 65.
        tau_Ds : TYPE, optional
            DESCRIPTION. The default is 85.

        Returns
        -------
        None.

        """
        super().__init__(
            density, young, shear, poisson, 
            Rm, Re, 
            sigma_ZDw, sigma_ZDs, 
            sigma_Bw, sigma_Bs, 
            tau_Dw, tau_Ds)