import pickle
import matplotlib.pyplot as plt

with open('/home/lsy/test/new/high_values_li.pkl', 'rb') as f:
    high_values = pickle.loads(f.read())

with open('/home/lsy/test/new/distance_li.pkl', 'rb') as f:
    distance = pickle.loads(f.read())

fig = plt.figure()
plt.subplot(1,2,1)
plt.scatter(distance[:-1], high_values[:-1])
plt.title("linear")

with open('/home/lsy/test/new/high_values_mlp.pkl', 'rb') as f:
    high_values_mlp = pickle.loads(f.read())

with open('/home/lsy/test/new/distance_mlp.pkl', 'rb') as f:
    distance_mlp = pickle.loads(f.read())

plt.subplot(1,2,2)
plt.scatter(distance_mlp[:-1], high_values_mlp[:-1])
plt.title("Mlp")



plt.show()
fig.savefig('/home/lsy/test/new/value_com.png')