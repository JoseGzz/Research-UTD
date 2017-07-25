import matplotlib.pyplot as plt
import networkx as nx
G=nx.dodecahedral_graph()
print("hi")
nx.draw(G)  # networkx draw()
plt.draw()  # pyplot draw()
plt.pause(10)