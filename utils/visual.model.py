import torch

# 加载模型
model = torch.load('logs/log_dist_lr0.005_bs16/checkpoint_dist.tar', map_location=torch.device('cuda:0'))

# 查看模型的键
print(type(model.keys()))
print(model['epoch'])
for i in range(3, 4):
    print(i)
# print(type(model['model_state_dict']['grasp_generator.tolerance.bn2.running_var']))
# print(model['model_state_dict']['grasp_generator.tolerance.bn2.running_var'].shape)
# print(model['model_state_dict']['grasp_generator.operation.conv3.weight'].shape)
# print(model['model_state_dict'].keys())

# # 查看模型的各个部分
# for key in model.keys():
#     print(key, model[key])