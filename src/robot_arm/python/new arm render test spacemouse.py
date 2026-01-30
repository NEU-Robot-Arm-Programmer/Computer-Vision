import numpy as np
import math
import trimesh
import pyrender 
from updated_kinematics_IK import *
import time
import serial
from datetime import datetime, timedelta
import pyspacemouse

L1 = 76.459
L2 = 250
L3 = 200
L31 = 40 #this is the length from the shoulder to offset the upper arm link 
L32 = 32.5 #this is the offset from the lower arm to the elbow
L6 = 50 

x = 350
y = 100
z = 250

a = 0
b = 0
g = 0

position_updated = False
update_timeout = 20
gripper_status = False

q1_max = 1000
q2_max = 400
q3_max = 400 
q4_max = 800
q5_max = 400
q6_max = 400

max_speeds = np.array([q1_max,q2_max,q3_max,q4_max,q5_max,q6_max])

#below are the number of steps in a full rotation for each joint
q1_steps = 200*25.500
q2_steps = 800*5.333333
q3_steps = 400*16.4706
q4_steps = 800*10.00
q5_steps = 800*4.8000
q6_steps = 800*4.8000

#steps to deg = 360/q_steps

steps = np.array([q1_steps,q2_steps,q3_steps,q4_steps,q5_steps,q6_steps])

viewer_refresh = 30 #pretty sure this does noting 

def translate(x,y,z): #makes a 4x4 3D affine transformation matrix with just pure translation
    output = np.eye(4)
    output[0][3] = x
    output[1][3] = y
    output[2][3] = z
    return output

def rotate(x_theta,y_theta,z_theta): #makes a 4x4 3D affine transformation matrix with just pure rotation, can be multiplied with a translation matrix
    output = np.eye(4)
    x_theta = to_rad(x_theta)
    y_theta = to_rad(y_theta)
    z_theta = to_rad(z_theta)
    rot_x = np.array([[1.0,0.0,0.0],[0.0,np.cos(x_theta),-1.0*np.sin(x_theta)],[0.0,np.sin(x_theta),np.cos(x_theta)]])
    rot_y = np.array([[np.cos(y_theta),0.0,np.sin(y_theta)],[0.0,1.0,0.0],[-1.0*np.sin(y_theta),0.0,np.cos(y_theta)]])
    rot_z = np.array([[np.cos(z_theta),-1.0*np.sin(z_theta),0.0],[np.sin(z_theta),np.cos(z_theta),0.0],[0.0,0.0,1.0]])
    rot_final = np.matmul(rot_z,rot_y)
    rot_final = np.matmul(rot_final,rot_x)
    for i in range(3):
        for j in range(3):
            output[i][j] = rot_final[i][j]
    return output

def scale(x,y,z):
    output = np.eye(4)
    output[0][0] = x
    output[1][1] = y
    output[2][2] = z
    return output

def to_deg(theta):
    output = theta*(180.0/np.pi)
    return output

def to_rad(theta):
    output = (theta*np.pi)/180.0
    return output 

def rot_x(theta):
    rot_x = np.array([[1.0,0.0,0.0],[0.0,np.cos(theta),-1.0*np.sin(theta)],[0.0,np.sin(theta),np.cos(theta)]])
    return rot_x

def rot_y(theta):
    rot_y = np.array([[np.cos(theta),0.0,np.sin(theta)],[0.0,1.0,0.0],[-1.0*np.sin(theta),0.0,np.cos(theta)]])
    return rot_y

def rot_z(theta):
    rot_z = np.array([[np.cos(theta),-1.0*np.sin(theta),0.0],[np.sin(theta),np.cos(theta),0.0],[0.0,0.0,1.0]])
    return rot_z

def to_deg(theta):
    output = theta*(180.0/np.pi)
    return output

def to_rad(theta):
    output = (theta*np.pi)/180.0
    return output 

def magnitude(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def vect_angle(v1,v2):
    cosang = np.dot(v1, v2)
    sinang = np.cross(v1, v2)
    normal = sinang/magnitude(sinang)
    #input = sinang/cosang
    return np.arctan2(np.dot(sinang,normal),cosang)


def position_segments(q1,q2,q3,q4,q5,q6): #posing works and the arm stays together, inverse kinematics are busted though
    pos_target = rotate(a,b,g) + translate(x,y,z) -np.eye(4)
    
    pos_ground = rotate(90,0,0) #place the links in their home positions
    pos_base = rotate(0,0,to_deg(q1))
    pos_shoulder = rotate(0,0,0)
    pos_upper_arm = translate(0,0,0)
    pos_elbow = rotate(0,0,0)
    pos_lower_arm = rotate(0,0,0)
    pos_wrist = rotate(0,0,0)
    
    q2 = -np.pi/2 -q2 #based on the way the meshes were saved the shoulder angle needs to be adjusted

    pos_shoulder = np.matmul(pos_base,rotate(0,to_deg(q2),0)) + translate(0,0,L1)-np.eye(4)

    pos_upper_arm = np.matmul(pos_shoulder,rotate(to_deg(q3),0,0)) + translate(L31*np.cos(q2)*np.cos(q1),L31*np.cos(q2)*np.sin(q1), L31*np.sin(-q2)) - np.eye(4)

    pos_elbow = pos_upper_arm + translate((L2-L31)*np.cos(q2)*np.cos(q1),(L2-L31)*np.cos(q2)*np.sin(q1),(L2-L31)*np.sin(-q2)) - np.eye(4)

    pos_elbow = np.matmul(pos_elbow,rotate(0,to_deg(q4),0))

    pos_lower_arm =  np.matmul(pos_elbow,rotate(to_deg(q5),0,0))

    lower_arm_vector = np.matmul(pos_lower_arm[:3,:3],np.array([1,0,0]))
    
    lower_arm_vector = lower_arm_vector#/np.linalg.norm(lower_arm_vector)

    wrist_xyz = (L3)*lower_arm_vector

    pos_wrist = np.matmul(pos_lower_arm,rotate(0,to_deg(q6),0)) + translate(wrist_xyz[0],wrist_xyz[1],wrist_xyz[2]) - np.eye(4)

    output = np.array([pos_ground,pos_base,pos_shoulder,pos_upper_arm,pos_elbow,pos_lower_arm,pos_wrist,pos_target])
    return(output)

def map(value, fromMin, fromMax, toMin, toMax):
    # Figure out how 'wide' each range is
    toSpan = toMax - toMin
    fromSpan = fromMax - fromMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - fromMin) / float(fromSpan)

    # Convert the 0-1 range into a value in the right range.
    return toMin + (valueScaled * toSpan)

def moition_scaler(current_joints,target_joints): #i think the goal of this was to output a speed that each joint could take in a certain move so they all arrived together
    #A = joints#np.array([joint_a[0],joint_a[1],joint_a[2],joint_a[3],joint_a[4],joint_a[5]]) #can this just be the input array? or is 'joint_a' not already a numpy array
    #B = inverse_kinematics(point[0],point[1],point[2],point[3],point[4],point[5]) #this will return a array of joint positions to get to point B
    diff = (current_joints-target_joints)*steps #take the goal and subtract the joints to find the angular distance and then get that number in steps (don't know if thats doing the steps part right)
    #print(diff)
    time = diff/max_speeds #find the time it will take each joint to complete this move using max speed for each
    move_time = time.max() #find the longest time of those joints and use that as the total time for the move 
    scaled_speeds = diff/move_time #scale the move to take the max amount of time (don't know if this is scaling correctly)
    #print(scaled_speeds)
    return scaled_speeds #output the scaled speeds


ground = trimesh.load(r"C:\Users\epfis\Documents\NU robotics\dummy geometry for renderer\dummy arm ground.stl")
ground.visual.face_colors = np.multiply(np.ones(ground.faces.shape),np.array([.75,.75,.75]))
ground_mesh = pyrender.Mesh.from_trimesh(ground, smooth = False)

base = trimesh.load(r"C:\Users\epfis\Documents\NU robotics\dummy geometry for renderer\dummy arm base.stl")
base.visual.face_colors = np.multiply(np.ones(base.faces.shape),np.array([.75,.75,.75]))
base_mesh = pyrender.Mesh.from_trimesh(base, smooth = False)

shoulder = trimesh.load(r"C:\Users\epfis\Documents\NU robotics\dummy geometry for renderer\dummy arm shoulder.stl")
shoulder.visual.face_colors = np.multiply(np.ones(shoulder.faces.shape),np.array([.75,.75,.75]))
shoulder_mesh = pyrender.Mesh.from_trimesh(shoulder, smooth = False)

upper_arm = trimesh.load(r"C:\Users\epfis\Documents\NU robotics\dummy geometry for renderer\dummy arm upper arm.stl")
upper_arm.visual.face_colors = np.multiply(np.ones(upper_arm.faces.shape),np.array([.75,.75,.75]))
upper_arm_mesh = pyrender.Mesh.from_trimesh(upper_arm, smooth = False)

elbow = trimesh.load(r"C:\Users\epfis\Documents\NU robotics\dummy geometry for renderer\dummy arm elbow.stl")
elbow.visual.face_colors = np.multiply(np.ones(elbow.faces.shape),np.array([.75,.75,.75]))
elbow_mesh = pyrender.Mesh.from_trimesh(elbow, smooth = False)

lower_arm = trimesh.load(r"C:\Users\epfis\Documents\NU robotics\dummy geometry for renderer\lower arm.stl")
lower_arm.visual.face_colors = np.multiply(np.ones(lower_arm.faces.shape),np.array([.75,.75,.75]))
lower_arm_mesh = pyrender.Mesh.from_trimesh(lower_arm, smooth = False)

wrist = trimesh.load(r"C:\Users\epfis\Documents\NU robotics\dummy geometry for renderer\dummy arm hand.stl")
wrist.visual.face_colors = np.multiply(np.ones(wrist.faces.shape),np.array([.75,.75,.75]))
wrist_mesh = pyrender.Mesh.from_trimesh(wrist, smooth = False)

#target = trimesh.creation.uv_sphere(radius = 10)
target = trimesh.creation.axis(origin_size = 6.0, origin_color = [1,.5,0])
#target.visual.face_colors = np.multiply(np.ones(target.faces.shape),np.array([1,0,0]))
target_mesh = pyrender.Mesh.from_trimesh(target, smooth = False)

origin = trimesh.creation.axis(origin_size = 10.0, origin_color = [1,.5,0])
origin_mesh = pyrender.Mesh.from_trimesh(origin, smooth = False)

light = pyrender.PointLight(color=[0.5, 1.0, 0.5], intensity=5.0)
cam = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0, znear=0.05, zfar=10000.0)


no = pyrender.Node(mesh=origin_mesh)
nground = pyrender.Node(mesh=ground_mesh, matrix=np.eye(4))
nbase = pyrender.Node(mesh=base_mesh)
ntarget = pyrender.Node(mesh=target_mesh, matrix = translate(500,0,300))

nshoulder = pyrender.Node(mesh=shoulder_mesh)
nupper_arm = pyrender.Node(mesh=upper_arm_mesh)
nelbow = pyrender.Node(mesh=elbow_mesh)
nlower_arm = pyrender.Node(mesh=lower_arm_mesh)
nwrist = pyrender.Node(mesh=wrist_mesh)


nl = pyrender.Node(light=light, matrix=np.eye(4))
nc = pyrender.Node(camera=cam, matrix=rotate(45,0,-35) + translate(0,-250,500) -np.eye(4))

scene = pyrender.Scene(ambient_light=[0.12, 0.12, 0.12],bg_color=[.9,.9,.9])

scene.add(origin_mesh)
scene.add_node(nground)
scene.add_node(nbase)
scene.add_node(ntarget)

scene.add_node(nshoulder)

scene.add_node(nupper_arm)
scene.add_node(nelbow)
scene.add_node(nlower_arm)
scene.add_node(nwrist)


scene.add_node(nl)
scene.add_node(nc)


coords_caption = {
  'text': f"X={x}  Y={y}  Z={z}  a={a}  b={b}  g={g}",
  'location': 'south',
  'font_name': "OpenSans-Regular",
  'font_pt': 18,
  'color': None,
  'scale': 1.0
}
joint_caption = {
  'text': f"q1={0}  q2={0}  q3={0}  q4={0}  q5={0}  q6={0}",
  'location': 'north',
  'font_name': "OpenSans-Regular",
  'font_pt': 18,
  'color': None,
  'scale': 1.0
}

captions = [coords_caption,joint_caption]
#there is an issue with captions and pyrender/opengl on windows 11, don't care enough to look into it rn
viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread= True, refresh_rate = viewer_refresh, window_title = '6-Axis Robot Visualizer',fullscreen = True)#, caption = captions)

time.sleep(0.5)


calibrated = False

success = pyspacemouse.open()

done = False
while not done:
    if viewer.is_active == False:
        done=True
    spacemouse = pyspacemouse.read()
    x = x + spacemouse.y
    y = y + -spacemouse.x
    z = z + spacemouse.z

    a = a + spacemouse.roll*.2
    b = b + spacemouse.pitch*.2
    g = g + spacemouse.yaw*.2
    

    angles = IK(x,y,z,a,b,g,L1,L2,L3,L6)
    poses = position_segments(angles[0],angles[1],angles[2],angles[3],angles[4],angles[5])

    coords_caption['text'] = f"X={x}  Y={y}  Z={z}  a={a}  b={b}  g={g}"
    joint_caption['text'] = f"q1={angles[0]}  q2={angles[1]}  q3={angles[2]}  q4={angles[3]}  q5={angles[4]}  q6={angles[5]}"

    #target_pos = prog_array[j]
    #poses = position_segments(temp_angle[0],temp_angle[1],temp_angle[2],temp_angle[3],temp_angle[4],temp_angle[5])

    viewer.render_lock.acquire()
    scene.set_pose(nground, pose=poses[0])
    scene.set_pose(nbase, pose = poses[1])
    scene.set_pose(nshoulder, pose=poses[2])

    scene.set_pose(nupper_arm, pose=poses[3])
    scene.set_pose(nelbow, pose = poses[4])
    scene.set_pose(nlower_arm, pose=poses[5])
    scene.set_pose(nwrist, pose=poses[6])

    scene.set_pose(ntarget,pose = poses[7])

    viewer.render_lock.release()


