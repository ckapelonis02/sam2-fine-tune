from training_helper import read_dataset, read_batch

data_size = 2000
data_dict = read_dataset("2k-hd-cropped/i", "2k-hd-cropped/m", list(range(1,data_size+1)))
points_num = []
for i in range(data_size):
    _, x, _, _ = read_batch(data_dict, i, -1)
    points_num.append(len(x))

print(f"max: {max(points_num)}")
print(f"min: {min(points_num)}")
print(f"avg: {sum(points_num)/len(points_num)}")
