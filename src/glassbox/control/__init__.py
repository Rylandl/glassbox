"""Closed-loop control: planning, command supervision, and identification.

The layer has one seam. :mod:`glassbox.control.plan` declares ``PlanModel``,
which is everything a solver knows about a model;
:mod:`glassbox.control.solver` optimizes over one; and
:mod:`glassbox.control.fitted` is the adapter that presents a fitted dynamics
belief as one. The recursive identifier's belief will target the same protocol
when the dual-control controller is resynced.
"""
