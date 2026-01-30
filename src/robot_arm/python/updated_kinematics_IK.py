import numpy as np

'''
x = 100
y = 300
z = 150

a = 0
b = 91 #setting these to 90 causes something to go to zero and it breaks 89 or 91 are ok though 
g = 0

#lengths of the links in the robot (regardless of links that twist relative to one another coaxially)
L0 = 50
L1 = 250 
L2 = 200
L3 = 50
'''

def deg_to_rad(x):
    rad = (x*(np.pi/180.0))
    return rad

def rad_to_deg(x):
    deg = (x*180.0)/np.pi
    return deg

def rotate(x_angle,y_angle,z_angle): #makes a rotation matrix which rotates a vector according to the input angles in the order x-->y-->z
    rx = np.array([[1,0,0],[0,np.cos(x_angle),np.sin(-x_angle)],[0,np.sin(x_angle),np.cos(x_angle)]])
    ry = np.array([[np.cos(y_angle),0,np.sin(y_angle)],[0,1,0],[np.sin(-y_angle),0,np.cos(y_angle)]])
    rz = np.array([[np.cos(z_angle),np.sin(-z_angle),0],[np.sin(z_angle),np.cos(z_angle),0],[0,0,1]])
    r = rz.dot(ry).dot(rx)
    return r

def find_elbow_points(wrist_point,wrist_normal,forearm_length,upper_arm_length,shoulder_pos):
    rho =  np.dot((wrist_point-shoulder_pos),wrist_normal)/np.linalg.norm(wrist_normal) #scalar distance from center of shoulder to the center of the intersection circle
 
    intersection_circle_center = shoulder_pos + np.dot(rho,wrist_normal/np.linalg.norm(wrist_normal))
    intersection_circle_radius = np.sqrt(upper_arm_length**2-np.abs(rho)**2)
    circle_circle_dist = np.linalg.norm(intersection_circle_center-wrist_point)

    intersection_point_x = (circle_circle_dist**2 - forearm_length**2 + intersection_circle_radius**2)/(2*circle_circle_dist)
    
    y1 = np.sqrt((intersection_circle_radius**2)-(intersection_point_x**2)) #this stuff is correct, the transformation onto the base frame is not

    center_wrist_vect = wrist_point - intersection_circle_center

    y_vect = np.cross(center_wrist_vect,wrist_normal)
    
    point_x = intersection_circle_center + intersection_point_x*(center_wrist_vect/np.linalg.norm(center_wrist_vect))

    point1 = point_x + np.dot(y1,(y_vect/np.linalg.norm(y_vect)))
    point2 = point_x - np.dot(y1,(y_vect/np.linalg.norm(y_vect)))
     
    possible_intersection_points = np.array([point1,point2])
    return possible_intersection_points

def IK(x,y,z,a,b,g,L0,L1,L2,L3):
    initial_target_pos = np.array([x,y,z])
    base_pos = np.array([0,0,L0])
    end_eff_start_dir = np.array([1,0,0])

    end_eff_vect = np.matmul(rotate(deg_to_rad(a),deg_to_rad(b),deg_to_rad(g)),end_eff_start_dir)
    wrist_normal_vect = np.matmul(rotate(deg_to_rad(a),deg_to_rad(b),deg_to_rad(g)),np.array([0,1,0])) #rotation from home position where wrist is aligned with y axis
    #wrist_tan_vect = np.matmul(rotate(deg_to_rad(a),deg_to_rad(b),deg_to_rad(g)),np.array([0,0,1]))    

    wrist_point = initial_target_pos - (L3*(end_eff_vect)/np.linalg.norm(end_eff_vect))
   
    intersection_points = find_elbow_points(wrist_point,wrist_normal_vect,L2,L1,base_pos)
    
    
    elbow_point =  intersection_points[0]
    #print(elbow_point)


    #wrist_point = initial_target_pos - (L3*(end_eff_vect)/np.linalg.norm(end_eff_vect))
    upper_arm_vect = (elbow_point-np.array([0,0,L0]))/np.linalg.norm(elbow_point-np.array([0,0,L0]))
    lower_arm_vect = (wrist_point - elbow_point)/(np.linalg.norm(wrist_point-elbow_point))

    axis_4 = np.arccos((np.dot(upper_arm_vect,lower_arm_vect))/(np.linalg.norm(upper_arm_vect)*np.linalg.norm(lower_arm_vect)))

    axis_6 = np.arctan2(np.dot((np.cross(L2*lower_arm_vect,L3*end_eff_vect)),wrist_normal_vect),np.dot(L2*lower_arm_vect,L3*end_eff_vect))

    axis_1 = np.arctan2(elbow_point[1],elbow_point[0])
    axis_2 = -np.pi/2 +np.arctan2(elbow_point[2]-L0,np.sqrt((elbow_point[0]**2)+(elbow_point[1]**2)))
    

    desired_rotated_elbow_axis = np.cross(lower_arm_vect,upper_arm_vect)
    desired_rotated_elbow_axis = desired_rotated_elbow_axis/np.linalg.norm(desired_rotated_elbow_axis)

    desired_rotated_wrist_axis = np.cross(end_eff_vect,lower_arm_vect)
    desired_rotated_wrist_axis = desired_rotated_wrist_axis/np.linalg.norm(desired_rotated_wrist_axis)


    shoulder_vect = np.cross(base_pos,upper_arm_vect)
    shoulder_vect = shoulder_vect/np.linalg.norm(shoulder_vect)

    elbow_vect = np.cross(upper_arm_vect,lower_arm_vect)
    elbow_vect = elbow_vect/np.linalg.norm(elbow_vect)

    axis_3 =  np.arctan2(np.dot(np.cross(shoulder_vect,elbow_vect),upper_arm_vect),np.dot(shoulder_vect,elbow_vect)) 
    if(axis_3>np.pi):
        axis_3 = axis_3 - (2*np.pi)

    wrist_vect = np.cross(lower_arm_vect,end_eff_vect)
    wrist_vect = wrist_vect/np.linalg.norm(wrist_vect)

    axis_5 =  -np.arctan2(np.dot(np.cross(wrist_vect,elbow_vect),lower_arm_vect),np.dot(wrist_vect,elbow_vect)) + np.pi

    if(axis_5>np.pi):
        axis_5 = axis_5 - (2*np.pi)

    if (np.abs(axis_5)>(np.pi/2)):
        if(axis_5>0):
            axis_5 = np.pi + axis_5
        if(axis_5<0):
            axis_5 = np.pi + axis_5

    if(axis_5>(2*np.pi)):
        axis_5 = np.pi - axis_5 
    

    return np.array([axis_1,axis_2,axis_3,axis_4,axis_5,axis_6])

'''
x = 100
y = 300
z = 150

a = 0
b = 91 #setting these to 90 causes something to go to zero and it breaks 89 or 91 are ok though 
g = 0

#lengths of the links in the robot (regardless of links that twist relative to one another coaxially)
L0 = 50
L1 = 250 
L2 = 200
L3 = 50
'''

#print(rad_to_deg(IK(350,0,150,52,61,0,76.459,250,200,50)))