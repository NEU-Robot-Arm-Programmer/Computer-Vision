"""
Notes:
    This math is based on a 6-dof arm, however even though our arm is 5-dof it has the same configuration (ZYYXY) leading up to the wrist 
    Since the configuration is the same, the math should be valid, but there will be orientations that the end effector cannot be placed in
    ex: if the wrist has to rotate out of the (xz) plane and but also be parallel with the ground (xy) plane
    Regardless of orientation I *think* that the end effector will still be able to reach whatever position we set, it just may be rotated in that position
    If the gripper was for example a pin or cylinder, the tip would be in the right place and this problem would not be noticeable 
    I think this problem will only come into play when we want to orient the gripper to pick items up that exist outside of the (xz) plane, so it still a good starting point
"""

import numpy as np
"""
x = 300.0   #position of the end effector with respect to the center of the base of the robot (in mm)
y = 00.0
z = 300.0
a = 0.0    #orientation of the end effector (in degrees)
b = -90.0
g = 0.0
"""
L1 = 241.52
L2 = 221.12
L3 = 225.5
L6 = 168.5

def rot_x(theta): #create a rotation matrix to rotate about x
    rot_x = np.array([[1.0,0.0,0.0],[0.0,np.cos(theta),-1.0*np.sin(theta)],[0.0,np.sin(theta),np.cos(theta)]])
    return rot_x

def rot_y(theta): #create a rotation matrix to rotate about y
    rot_y = np.array([[np.cos(theta),0.0,np.sin(theta)],[0.0,1.0,0.0],[-1.0*np.sin(theta),0.0,np.cos(theta)]])
    return rot_y

def rot_z(theta): #create a rotation matrix to rotate about z
    rot_z = np.array([[np.cos(theta),-1.0*np.sin(theta),0.0],[np.sin(theta),np.cos(theta),0.0],[0.0,0.0,1.0]])
    return rot_z

def to_deg(theta): 
    output = theta*(180.0/np.pi)
    return output

def to_rad(theta):
    output = (theta*np.pi)/180.0
    return output 

def xyz_adjuster(x,y,z,alpha,beta):  #function that will adjust the target position for IK of the lower 3 joints based on the knowledge that the wrist link has a known length and orientation determined by (alpha,beta,gamma)
    x_a = x - L6*np.cos(-1.0*alpha)*np.cos(beta)
    y_a = y - L6*np.sin(-1.0*alpha)*np.cos(beta)
    z_a = z - L6*np.sin(beta) 
    return (x_a,y_a,z_a)

def lower_joint_IK(x,y,z):  #function to calculate the IK of the lower joints so the base of the wrist is in the right place
    solutions = [] #this list serves no purpose, the output of the function is nested in here for no reason but I don't want to change it rn
    R = np.sqrt((x*x)+(y*y)+((z-L1)*(z-L1)))
    radial_intersect = ((L3*L3)-(L2*L2)-(R*R))/(-2*L2) 

    q1 = np.arctan2(y,x)  
    q2 = np.arccos(((L2*L2)+(R*R)-(L3*L3))/(2*L2*R)) + (np.arcsin((z-L1)/(R)))

    q3 = np.arccos((radial_intersect-L2)/L3)
    solutions.append((q1,q2,q3))
    return solutions

def inverse_kinematics(x,y,z,a,b,g):  #function to return the positions of all the joints based on the input position and orientation
    np.set_printoptions(precision=20)

    alpha = to_rad(a)
    beta = to_rad(b)
    gamma = to_rad(g)

    x_a = xyz_adjuster(x,y,z,alpha,beta)[0]
    y_a = xyz_adjuster(x,y,z,alpha,beta)[1]
    z_a = xyz_adjuster(x,y,z,alpha,beta)[2]

    RD = np.matmul(rot_y(beta),rot_x(gamma))
    RD = np.matmul(rot_z(alpha),RD)

    q1 = lower_joint_IK(x_a,y_a,z_a)[0][0]
    q2 = lower_joint_IK(x_a,y_a,z_a)[0][1]
    q3 = lower_joint_IK(x_a,y_a,z_a)[0][2]

    R3 = np.matmul(rot_z(-q1),rot_y(q2))
    R3 = np.matmul(R3,rot_y(-q3))

    R3_inv = np.transpose(R3)
    R_final = np.matmul(R3_inv,RD)

    sin_q4 = R_final[1,0]
    cos_q4 = -R_final[2,0]

    sin_q5 = np.sqrt(R_final[0,1]**2+R_final[0,2]**2)
    cos_q5 = R_final[0,0]

    sin_q6 = R_final[0,1]
    cos_q6 = R_final[0,2]

    q4 = np.arctan2(sin_q4,cos_q4)
    q5 = np.arctan2(sin_q5,cos_q5)
    q6 = np.arctan2(sin_q6,cos_q6)

    #print("q1: " + str(to_deg(q1)))
    #print("q2: " + str(to_deg(q2)))
    #print("q3: " + str(to_deg(q3*-1)))
    #print("q4: " + str(to_deg(q4)))
    #print("q5: " + str(to_deg(q5)))
    #print("q6: " + str(to_deg(q6)))
    #print("--------------------")

    return np.array([q1,q2,q3,q4,q5,q6])

if __name__=='__main__':

    joint_angles = inverse_kinematics(x,y,z,a,b,g)

    """
    print("q1: " + str(to_deg(joint_angles[0])))
    print("q2: " + str(to_deg(joint_angles[1])))
    print("q3: " + str(to_deg(joint_angles[2])))
    print("q4: " + str(to_deg(joint_angles[3])))
    print("q5: " + str(to_deg(joint_angles[4])))
    print("q6: " + str(to_deg(joint_angles[5])))
    print("--------------------")
    """
