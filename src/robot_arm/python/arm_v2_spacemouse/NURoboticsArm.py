#!/usr/bin/env python3
"""
NU Robotics Arm model for Peter Corke's Robotics Toolbox for Python.

This follows the general `roboticstoolbox.models.DH` pattern used in the
repository at petercorke/robotics-toolbox-python.

Important note:
The handwritten table appears to list link 3 with a variable in the
alpha-column and a constant -90 deg in the theta/phi-column:

    link 3: alpha = theta_3, a = 0, d = 0, phi = -90 deg

That is not directly representable by `DHRobot` using standard DH links,
since the toolbox expects the joint variable for a revolute joint to appear
in theta, not alpha.

For this file, I have used the most likely standard-DH interpretation:
link 3 is modeled as a revolute joint with a constant theta offset of -90 deg
and alpha = 0.  If your intended convention is different, link 3 will need to
be adjusted.
"""

from math import pi
import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH


class NURoboticsArm(DHRobot):
    """
    Class that models the NU Robotics arm using standard DH conventions.

    Parameters
    ----------
    L1, L2, L3, L4 : float
        Geometric parameters from the sketch, in meters.

    Notes
    -----
    Interpreted DH table used here:

    ======  ========  ========  ========  =====================
    Link    alpha     a         d         theta
    ======  ========  ========  ========  =====================
    1       0         0         L1        q1
    2       -pi/2     0         0         q2
    3       0         0         0         q3 - pi/2   (assumed)
    4       pi/2      0         L2        q4
    5       -pi/2     L3        0         q5
    6       pi/2      0         L4        q6
    ======  ========  ========  ========  =====================
    """

    def __init__(self, L1=.2106, L2=.29327, L3=.1395, L4=.259, L5 = .1):
        deg = pi / 180.0

        links = [
            RevoluteDH(
                d=L1,
                a=0,
                alpha=pi/2,
                qlim=[-180 * deg, 180 * deg],
            ),
            RevoluteDH(
                d=0.0,
                a=0.0,
                alpha=-pi / 2,
                offset = 0,
                qlim=[-120 * deg, 120 * deg],
            ),
            RevoluteDH(
                d=L2,
                a=0.0,
                alpha=-pi / 2,
                offset=0,
                qlim=[-160 * deg, 160 * deg],
            ),
            RevoluteDH(
                d=0.0,
                a=L3,
                alpha=-pi / 2,
                #offset=pi,
                qlim=[-163 * deg, 0 * deg],
            ),
            RevoluteDH(
                d=L4,
                a=0,
                alpha=pi/2,
                offset=0,
                qlim=[-160 * deg, 160 * deg],
            ),
            RevoluteDH(
                d=0,
                a=L5,
                alpha=0,
                offset=pi/2,
                qlim=[-90 * deg, 90 * deg],
            )
        ]

        super().__init__(
            links,
            name="NU Robotics Arm",
            manufacturer="Northeastern University",
        )

        self.qz = np.zeros(6)
        self.qr = np.array([0.0, -pi / 2, -pi / 2, 0.0, 0.0, 0.0,])

        self.addconfiguration("qz", self.qz)
        self.addconfiguration("qr", self.qr)


if __name__ == "__main__":
    robot = NURoboticsArm(L1=.2106, L2=.29327, L3=.1395, L4=.259, L5=.1)
    print(robot)
    print("\nZero pose FK:")
    print(robot.fkine(robot.qz))