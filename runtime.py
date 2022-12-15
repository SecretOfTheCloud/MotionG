import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.transform import Rotation as R
import normalize
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_frames = 1200

model = torch.load("./model.pth")
model.eval()

runtime_data = np.load('./runtime_data.npz')

# x=torch.tensor(np.zeros([1,316]).astype(np.float32))
# p=torch.tensor(np.zeros([1,12]).astype(np.float32))
# z=torch.tensor(np.zeros([1,2]).astype(np.float32))
x=torch.tensor(runtime_data['X'].astype(np.float32)).to(device)
p=torch.tensor(runtime_data['P'].astype(np.float32)).to(device)
z=torch.tensor(runtime_data['Z'].astype(np.float32)).to(device)
y=torch.tensor(np.zeros([1,306]).astype(np.float32)).to(device)
print(555555555555555555)
print(x)
print(p)
print(z)
print(555555555555555555)

offset_global = np.zeros([1,3])
offset_global_list = np.array([]).reshape((0,3))

rotation_root_global = R.from_euler('YXZ', [0,0,0], degrees=True)
rotation_root_global_list = []

bvh_data=np.array([]).reshape(0,72)


for frame in range(output_frames):
    bvh_line = []

    offset_global_list=np.append(offset_global_list,offset_global,axis=0)
    rotation_root_global_list.append(rotation_root_global)

    y=model(x,p,z)
    x, y, p, z = map(lambda x: x.to("cpu").detach().numpy(), [x,y,p,z])
    # print(222222222222)
    # print(frame)
    # print(p)
    # print(z)

    y=normalize.Run_denormalize_Y(y)
    x=normalize.Run_denormalize_X(x)

# update!
# update!!
# update!!!


    # phase
    
    # x[:,304:310]=y[:,294:300]*1/60 + x[:,304:310]
    # x[:,304:310]=(((y[:,294:300]+0.5)%1-0.5)+x[:,304:310]+y[:,300:306]/60)/2.0
    # x[:,304:310]=((y[:,294:300]+x[:,304:310]+y[:,300:306]/60)/2+0.5)%1-0.5
    x[:,304:310]=(x[:,304:310]+y[:,300:306]/60+0.5)%1-0.5


    print(x[:,304:310])
    # x[:,310:316]=y[:,300:306]

    # rotation
    rotation_next = y[:,182:294].reshape(28,4)
    rotation_next_list = []
    rotation_next_update_list = []
    for rotation in rotation_next:
        # print(rotation)
        rotation_next_list.append(R.from_quat(rotation))

    for rotation in rotation_next_list:
        rotation_next_update_list.append(R.inv(rotation_next_list[0])*rotation)

    rotation_1 = np.array([]).reshape((0,4))

    for rotation in rotation_next_update_list:
        rotation_1 = np.append(rotation_1,rotation.as_quat())

    x[:,192:304]=rotation_1.reshape(1,112)


    #记录bvh数据
    bvh_line=list(offset_global.reshape(-1))
    bvh_line+=list(rotation_root_global.as_euler('YXZ', degrees=True))
    for joint in range(28):
        if joint not in [0,7,12,17,22,27]:
            bvh_line+=list(rotation_next_list[joint].as_euler('YXZ', degrees=True))

    bvh_data=np.append(bvh_data,np.array(bvh_line).reshape(1,72),axis=0)

    # global
    offset_global = offset_global+rotation_root_global_list[-1].as_matrix()@(y[:,14:17].reshape(-1))
    rotation_root_global = rotation_root_global*rotation_next_list[0]

    

    # position
    for i in range(28):
        x[:,24+3*i:27+3*i] = R.inv(rotation_next_list[0]).as_matrix()@(y[:,14+3*i:17+3*i]-y[:,14:17]).reshape(-1)

    # velocity
    x[:,108:192]=y[:,14:98]-x[:,24:108]

    #轨迹
    for numbre, intervalle in enumerate([-49,-39,-29,-19,-9]):
        if frame+intervalle>=0:
            # print((offset_global_list[frame+intervalle]-offset_global).shape)
            courrant = R.inv(rotation_root_global).as_matrix()@(offset_global_list[frame+intervalle]-offset_global).reshape(-1)
            x[:,2*numbre]=courrant[0]
            x[:,2*numbre+1]=courrant[2]

    x[:,10:24]=y[:,0:14]

    x = normalize.Run_normalize_X(x)
    p = copy.deepcopy(x[:,-12:])

    x, y, p, z = map(lambda x: torch.from_numpy(x).to(device), [x,y,p,z])

np.savetxt('./bvh_data.txt', np.c_[bvh_data],
 fmt='%f',delimiter='\t')
        
    




y=model(x,p,z)
print(y)