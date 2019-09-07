import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
sns.set(style="darkgrid")

N = 700

# scratch data
csv_data = pd.read_csv('/home/wr1/rllab/data/local/hier25-8y/hier25_8y_4scale_50agg_400pl_PREAnt-snn_005MI_5grid_6latCat_bil_0020_seed0_2019_04_10_10_31_56/progress.csv')
csv_data[' '] = 'PPO'
csv_data = csv_data.drop([0, 1])
print(csv_data.shape)  # (189, 9)
num_row = len(csv_data.index)
csv_data['episodes'] = range(num_row)


# csv_data1 = pd.read_csv('scratch2.csv')
# csv_data1[' '] = 'PPO'
# print(csv_data1.shape)  # (189, 9)
# num_row1 = len(csv_data1.index)
# csv_data1['episodes'] = range(num_row1)

# csv_data2 = pd.read_csv('scratch3.csv')
# csv_data2[' '] = 'PPO'
# print(csv_data2.shape)  # (189, 9
# num_row2 = len(csv_data2.index)
# csv_data2['episodes'] = range(num_row2)

# # reuse data
# reuse_data = pd.read_csv('reuse1.csv')
# reuse_data[' '] = 'Our'
# num_row = len(reuse_data.index)
# reuse_data['episodes'] = range(num_row)
#
# reuse_data1 = pd.read_csv('reuse2.csv')
# reuse_data1[' '] = 'Our'
# num_row1 = len(reuse_data1.index)
# reuse_data1['episodes'] = range(num_row1)

# reuse_data2 = pd.read_csv('reuse3.csv')
# reuse_data2[' '] = 'Our'
# num_row2 = len(reuse_data2.index)
# reuse_data2['episodes'] = range(num_row2)
#
# reuse_data3 = pd.read_csv('reuse4.csv')
# reuse_data3[' '] = 'Our'
# num_row3 = len(reuse_data3.index)
# reuse_data3['episodes'] = range(num_row3)
# print(reuse_data3.head(N))

# df3 = pd.concat([csv_data.head(N), csv_data1.head(N), csv_data2.head(N),\
#                 reuse_data.head(N), reuse_data1.head(N),reuse_data2.head(N), reuse_data3.head(N)])
# df3.drop(['Wall time'], axis=1)
# df4 = df3.rename(columns={'Value':"episodic reward"})
# print(df4)

fig = plt.figure()
sns.lineplot(x='episodes', y="LossAfter_low",
             hue=' ',
             data=csv_data)

plt.show()
fig.savefig('high_lr.pdf')
