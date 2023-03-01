import torch


s0_grid_arr = torch.as_tensor([[ 0, 16, 32],
                               [48, 64, 80]])
s1_grid_arr = torch.as_tensor([[ 0,  2, 16, 18, 32, 34],
                               [ 8, 10, 24, 26, 40, 42],
                               [48, 50, 64, 66, 80, 82],
                               [56, 58, 72, 74, 88, 90]])
s2_grid_arr = torch.as_tensor([[ 0,  1,  2,  3, 16, 17, 18, 19, 32, 33, 34, 35],
                               [ 4,  5,  6,  7, 20, 21, 22, 23, 36, 37, 38, 39],
                               [ 8,  9, 10, 11, 24, 25, 26, 27, 40, 41, 42, 43],
                               [12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47],
                               [48, 49, 50, 51, 64, 65, 66, 67, 80, 81, 82, 83],
                               [52, 53, 54, 55, 68, 69, 70, 71, 84, 85, 86, 87],
                               [56, 57, 58, 59, 72, 73, 74, 75, 88, 89, 90, 91],
                               [60, 61, 62, 63, 76, 77, 78, 79, 92, 93, 94, 95]])
print('\nOrig:')
print(s0_grid_arr.shape)
print(s0_grid_arr)
print(s1_grid_arr.shape)
print(s1_grid_arr)
print(s2_grid_arr.shape)
print(s2_grid_arr)


s0_flat_arr = torch.reshape(s0_grid_arr, (-1,))
s1_flat_arr = torch.reshape(s1_grid_arr, (-1,))
s2_flat_arr = torch.reshape(s2_grid_arr, (-1,))
print('\nFlat:')
print(s0_flat_arr)
print(s1_flat_arr)
print(s2_flat_arr)


def expand_scale(tensor, scale, grid_size_x):
    r = torch.repeat_interleave(tensor, scale)
    r = torch.reshape(r, (-1, grid_size_x))
    r = torch.repeat_interleave(r, scale, 0)
    return r


expanded_s0_arr = expand_scale(s0_flat_arr, 4, 12)
expanded_s1_arr = expand_scale(s1_flat_arr, 2, 12)
print('\nRepeated:')
print(torch.flatten(expanded_s0_arr))
print(torch.flatten(expanded_s1_arr))
print(s2_flat_arr)

print('\nReshaped:')
print(expanded_s0_arr)
print(expanded_s1_arr)
print(s2_grid_arr)



