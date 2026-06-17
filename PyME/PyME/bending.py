#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 22:35:42 2026

@author: fabian
"""

"""
This Module contains Objects for solving bending problems of determined and overdetermined Systems.
A Beamsystem consists of multiple Nodes and Subsections. In general, a Beam with N Subsections has N+1 Nodes.

Here are a few examples from the book "technische Mechanik":

A Beam with two Nodes and one SubSegment.
linear Load distribution proportional to the length of the Beam.
>>> l, q0, E, I, x = sp.symbols("l q_0 E I x")

>>> n1 = SymNode(1, 0, E, I).set_as_boundarybearing()
>>> n2 = SymNode(2, l, E, I).set_as_boundarybearing().set_load(q0 * x / l)
>>> nodes = (n1, n2)

>>> s1 = SubSegment(n1, n2, 1)
>>> subsegments = (s1)

>>> subst = ((l, 100), (q0, -0.2), (E, 210_000), (I, 1_700_000))
>>> system = BeamSystem(subsegments, nodes).solve_system()
>>> system.plot_system(subst)


A Beam with four Nodes and three SubSegments.
A point load on the end of the first Segment and a constant load distribution on the last Segment.
>>> q, F, E, I, L = sp.symbols("q F E I L")
>>> n1 = SymNode(1, 0, E, I).set_as_boundarybearing()
>>> n2 = SymNode(2, L, E, I).set_force(F)
>>> n3 = SymNode(3, 1.5*L, E, I).set_as_midbearing()
>>> n4 = SymNode(4, 2*L, E, I).set_as_boundarybearing().set_load(q)
>>> nodes = (n1, n2, n3, n4)

>>> s1 = SubSegment(n1, n2, 1)
>>> s2 = SubSegment(n2, n3, 2)
>>> s3 = SubSegment(n3, n4, 3)
>>> segs = (s1, s2, s3)

>>> subs = ((q, -2), (F, -2500), (E, 210_000), (I, 170_000_000), (L, 500))
>>> system = BeamSystem(segs, nodes)
>>> system.solve_system()
>>> system.plot_system(subs)
"""

import sympy as sp
import numpy as np


class SymNode(object):
    def __init__(
        self,
        subscript: any,
        position: sp.Symbol,
        young: sp.Symbol,
        inertia: sp.Symbol
    ) -> None:
        """
        Nodes are connection points between Subsegments of the Beam.
        They serve as pickup points for outer forces and moments that act on the beam.
        A few bearing-types can be set onto the node. This generates different boundary- and transitional conditions.
        
        Methods:
            set_as_boundarybearing():
                Place a bearing on the position of the node.
                Only use on nodes that are on an outmost positions of the beam.
                It has the attributes: Q = unknown, M = 0, w' = unknown, w = 0
            
            set_as_midbearing():
                Place a bearing on the position of the node.
                The Node must be somwhere mid-beam.
                It has the attributes: Q = unknown, M = unknown, w' = 0, w = 0
            
            set_as_rigid():
                Fix the beam at the position of the node
                The attributes of the beam at this position are: Q = unknown, M = unknown, w' = 0, w = 0
                
            set_as_loose():
                Declare the beam as unhindered at the position of the node.
                Only use on nodes that are on an outmost positions of the beam.
                The attributes of the beam at this position are: Q = 0, M = 0, w' = unknown, w = unknown
                
            set_load(load, is_start, is_end):
                load (float|symbol): some definition of the force distribution.
                Can be a constant, a symbol or a symbolic pseudo function of x [N/mm]
                ONLY USE set_load ON THE END OF THE LOAD DISTRIBUTION!!!
                
            
            set_force(force, moment):
                force (float|symbol): some definition of a point force acting on the beam at the position of the node.
                    Can be a constant, a symbol or a symbolic pseudo function of x [N]
                moment (float|symbol): Some definition of a moment acting on the beam at the position of the node. 
                    Can be a constant, a symbol or a symbolic pseudo function of x [N/mm]

        Args:
            subscript (str | int): number of the node
            position (sp.Symbol): position of the node on the beam [mm]
            young (sp.Symbol): Elasticity modulus of the beam at position of the node [N/mm^2]
            inertia (sp.Symbol): inertial modulus of the beam at position of the node [mm^4]
        """
        self.x = sp.symbols("x")
        self.smbl = sp.symbols(f"N_{subscript}")
        self.position = position
        self.bc = np.array([1, 1, 1, 1])
        self.young = young
        self.inertia = inertia
        self.load = 0
        self.force = np.array([0, 0, 0, 0])
    
    
    def set_as_boundarybearing(self):
        self.bc = np.array([1, 0, 1, 0]).T
        return self
    
    
    def set_as_midbearing(self):
        self.bc = np.array(["unknown", 1, 1, 0]).T
        return self
    
    
    def set_as_rigid(self):
        self.bc = np.array([1, 1, 0, 0]).T
        return self
    
    
    def set_as_loose(self):
        self.bc = np.array([0, 0, 1, 1]).T
        return self
    
    
    def set_as_glider(self):
        self.bc = np.array([0, 1, 0, 1]).T
        return self
    
    
    def set_load(self, load):
        self.load = load
        return self
    
    
    def set_force(self, force=0, moment=0, phi=0, theta=0):
        self.force = np.array([force, moment, phi, theta])
        return self
    
    
    def _get_equations(self, x0, integration_constants):
        c1, c2, c3, c4 = integration_constants
        Q_x = sp.integrate(self.load, self.x) + c1
        M_x = sp.integrate(Q_x, self.x) + c2
        dw_x = (sp.integrate(M_x / (self.young * self.inertia), self.x)) + c3
        w_x = sp.integrate(dw_x, self.x) + c4
        Q_x, M_x, dw_x, w_x = Q_x.simplify(), M_x.simplify(), dw_x.simplify(), w_x.simplify()
        return [Q_x.subs(self.x, x0), M_x.subs(self.x, x0), dw_x.subs(self.x, x0), w_x.subs(self.x, x0)]


class SubSegment(object):
    def __init__(self,
        startpoint: SymNode,
        endpoint: SymNode,
        subscript: str|int
    ) -> None:
        self.startpoint = startpoint
        self.endpoint = endpoint
        self.forces = endpoint.force
        self.smbl = sp.symbols(f"S_{subscript}")
        self.integration_constants = sp.Matrix([sp.symbols(f"c_{subscript*4-3} c_{subscript*4-2} c_{subscript*4-1} c_{subscript*4}")])


    def _create_subLSE(self):
        self.length = self.endpoint.position - self.startpoint.position
        lh = self.startpoint._get_equations(0, self.integration_constants)
        rh = self.endpoint._get_equations(self.length, self.integration_constants)
        if self.startpoint.position == 0:
            self.lhs = np.array(lh) + self.startpoint.force
        else:
            self.lhs = np.array(lh)
        self.rhs = np.array(rh) + self.endpoint.force
        return self


    def _create_equations(self, constants, substitutions):
        equations = self.endpoint._get_equations(self.startpoint.x, self.integration_constants)
        zipped_subs = list(zip(constants, substitutions))
        for ind, eq in enumerate(equations):
            for sym, sub in zipped_subs:
                equations[ind] = equations[ind].subs(sym, sub)
        return equations


class BeamSystem(object):
    def __init__(self,
        elements: iter,
        nodes: iter
    ) -> None:
        try:
            elements[0]
            self.elements = elements
        except TypeError:
            self.elements = [elements for n in range(1)]
        try:
            nodes[0]
            self.nodes = nodes
        except TypeError:
            self.nodes = [nodes for n in range(1)]


    def solve_system(self):
        """
        Calculate the piecewise- functions of x for the following:
            Q(x): Shear force progression per Subsegment
            M(x): Moment progression per Subsegment
            w'(x): Inclination progression per subsegment
            w(x): Deflection progression per subsegment
            
        after solving the system, the symbolic functions can be accessed via the attribute: all_functions.
        Access the integration constants via the attribute: solution
        """
        for element in self.elements:
            element._create_subLSE()
        
        constants = np.hstack([element.integration_constants for element in self.elements])[0]
        
        start_eqs = np.delete(
            self.elements[0].lhs,
            [ind for ind, value in enumerate(self.elements[0].startpoint.bc) if value]
        )
        end_eqs = np.delete(
            self.elements[-1].rhs,
            [ind for ind, value in enumerate(self.elements[-1].endpoint.bc) if value]
        )
        
        self.system = np.hstack((
            sp.Eq(start_eqs[0], 0),
            sp.Eq(start_eqs[1], 0)
        ))
        try:
            lhs_eqs = np.hstack([el.lhs for el in self.elements[1:]])
            rhs_eqs = np.hstack([el.rhs for el in self.elements[:-1]])
            lhs_bc = np.hstack([el.startpoint.bc for el in self.elements[1:]])
            for ind, bc in enumerate(lhs_bc):
                if bc == "unknown":
                    pass
                elif not int(bc):
                    self.system = np.hstack((
                        self.system,
                        sp.Eq(lhs_eqs[ind], 0),
                        sp.Eq(0, rhs_eqs[ind])
                    ))
                elif int(bc):
                    self.system = np.hstack((
                        self.system,
                        sp.Eq(lhs_eqs[ind], rhs_eqs[ind])
                    ))
        except ValueError:
            lhs_eqs = np.array([el.lhs for el in self.elements[1:]])
            rhs_eqs = np.array([el.rhs for el in self.elements[:-1]])
            lhs_bc = np.array([el.startpoint.bc for el in self.elements[1:]])
            for ind, bc in enumerate(lhs_bc):
                if bc == "unknown":
                    pass
                elif not int(bc):
                    self.system = np.array((
                        self.system,
                        sp.Eq(lhs_eqs[ind], 0),
                        sp.Eq(0, rhs_eqs[ind])
                    ))
                elif int(bc):
                    self.system = np.array((
                        self.system,
                        sp.Eq(lhs_eqs[ind], rhs_eqs[ind])
                    ))

        self.system = np.hstack((
            self.system,
            sp.Eq(end_eqs[0], 0),
            sp.Eq(end_eqs[1], 0)
        ))

        self.solution = sp.solve(self.system, constants)
        assert isinstance(self.solution, dict), "no solution for integration constants found"
        assert len(self.solution) == len(constants), "not all constants have a solution"
        # assert not np.all([self.solution.get(key) for key in self.solution.keys()]), "all constants equal to zero"

        all_equations = []
        for element in self.elements:
            self.sub_equations = element._create_equations(
                self.solution.keys(),
                [self.solution.get(key) for key in self.solution.keys()]
            )
            all_equations.append(self.sub_equations)

        self.all_equations = np.array(all_equations)
        return self


    def plot_system(self, substitutions = {}):
        """
        assemble all functions needed for plotting.

        Args:
            substitutions (dict): substitute every key given with the corresponding value in all equations.

        returns:
            [[arrays], [callables], position x of Mbmax]
        """
        x = np.array([])
        y = [[], [], [], []]
        for ind, subset in enumerate(self.elements):
            el_end = subset.endpoint.position
            el_start = subset.startpoint.position
            for sym, sub in substitutions:
                if not subset.startpoint.position:
                    el_start = subset.startpoint.position
                else:
                    el_start = el_start.subs(sym, sub)
                el_end = el_end.subs(sym, sub)
                self.all_equations[ind][0] = self.all_equations[ind][0].subs(sym, sub)
                self.all_equations[ind][1] = self.all_equations[ind][1].subs(sym, sub)
                self.all_equations[ind][2] = self.all_equations[ind][2].subs(sym, sub)
                self.all_equations[ind][3] = self.all_equations[ind][3].subs(sym, sub)

            Qx = sp.lambdify(subset.startpoint.x, self.all_equations[ind][0])
            Mx = sp.lambdify(subset.startpoint.x, self.all_equations[ind][1])
            dwx = sp.lambdify(subset.startpoint.x, self.all_equations[ind][2])
            wx = sp.lambdify(subset.startpoint.x, self.all_equations[ind][3])

            x0 = np.linspace(0, float(el_end - el_start), 250)
            x = np.hstack((x, x0+el_start))

            if isinstance(Qx(x0), np.ndarray):
                y[0].append(Qx(x0))
            else:
                y[0].append(np.full_like(x0, Qx(x0)))
            
            if isinstance(Mx(x0), np.ndarray):
                y[1].append(Mx(x0))
            else:
                y[1].append(np.full_like(x0, Mx(x0)))
            
            if isinstance(dwx(x0), np.ndarray):
                y[2].append(dwx(x0))
            else:
                y[2].append(np.full_like(x0, dwx(x0)))
            
            if isinstance(wx(x0), np.ndarray):
                y[3].append(wx(x0))
            else:
                y[3].append(np.full_like(x0, wx(x0)))

        Q_x = np.hstack(y[0])
        M_x = np.hstack(y[1])
        dw_x = np.hstack(y[2])
        w_x = np.hstack(y[3])

        return [[Q_x, M_x, dw_x, w_x], [Qx, Mx, dwx, wx], x]
