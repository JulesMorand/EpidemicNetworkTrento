import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

#number of nodes (individuals)
N=1000
#generation of a graph (Here barabasi albert)
G = nx.generators.random_graphs.barabasi_albert_graph(N,1)
#G = nx.gnp_random_graph(N,4./N)
#ajacency matrix of the graph
A=nx.adjacency_matrix(G)
print(A.todense())
# initialisation of the vectors probability rates 
#(the same probabitity for each nodes)
ProbaS=np.full(N, 1.-2./N)
ProbaI=np.full(N, 2./N)
ProbaR=np.full(N, 0.)
#print(ProbaS)
#print(ProbaI)
#print(ProbaR)	
# initialisation of random vector X containing the state of each nodes
# (weighted sampling with replacement)
X=random.choices(['S', 'I', 'R'], [ProbaS[0], ProbaI[0], ProbaR[0]], k=N)
#print(X)
# view of the graph with colors in function of the state
color_map=[] 
#such that the graphs are plotted in the same position
seed = 31 
#pos=nx.spring_layout(G,seed=seed)
#pos = nx.kamada_kawai_layout(G)
#pos=nx.planar_layout(G)
# plt.figure(figsize=(12,8))
# for i in X:
#     if i =='S':
#         color_map.append('blue')
#     elif i=='I': 
#         color_map.append('red') 
#     else:      
#     	color_map.append('green') 
# nx.draw(G,pos=pos,node_color=color_map,node_size=40)
#plt.pause(0.1)
#plt.close()

# Definition of the 'delta' probability vectors
deltaS=np.zeros(N)
deltaI=np.zeros(N)
deltaR=np.zeros(N)
prod=np.full(N, 1.)
#List of neigbourg list
Neig=[]
for i in range(N):
	Neig_i=[]
	for j in range(N):
		print(i,j,end='\r')
		if A[i,j]==1: Neig_i.append(j)
	Neig.append(Neig_i)	
mu=0.1 #Probability of recover in one time step : Stay infected in mean 1/mu time steps
beta=0.7 #Probability to be infected by a neigbourg in one time step : Will infect in mean in 1/beta time steps

# Equation of evolution for the vectors probability rates
T=1000 #Total number of time steps
rhoS=np.zeros(T)
rhoI=np.zeros(T)
rhoR=np.zeros(T)
R=np.zeros(T-1)
for t in np.arange(T) :
	print('t=',t,end='\r')
	for i in np.arange(N) :
		if X[i]=='S' : 
			deltaS[i]=1.
			rhoS[t]+=1./N
		else : deltaS[i]=0.
		if X[i]=='I' : 
			deltaI[i]=1.
			rhoI[t]+=1./N 
		else : deltaI[i]=0.
		if X[i]=='R' : 
			deltaR[i]=1.
			rhoR[t]+=1./N
		else : deltaR[i]=0.
		prod[i]=1
		for j in Neig[i]:
			prod[i]=prod[i]*(1-beta*deltaI[j])

		ProbaS[i]=deltaS[i]*prod[i]
		ProbaI[i]=(1-mu)*deltaI[i]+deltaS[i]*(1-prod[i])
		ProbaR[i]=deltaR[i]+mu*deltaI[i] 

		X[i]=random.choices(['S', 'I', 'R'], [ProbaS[i], ProbaI[i], ProbaR[i]],k=1)[0]
#	print('ProbaS=', ProbaS)
#	print('ProbaI=', ProbaI)
#	print('ProbaR=', ProbaR)
#	print('X=',X)
#Computation reproduction number:
#	if t>=1 : R[t-1]=(rhoI[t]-rhoI[t-1])/rhoI[t-1] # #ofnewcase at t/#infected at t-1
# Plots of the sucessive states of the graphs.
	# color_map=[]
	# for i in X:
	# 	if i =='S':color_map.append('blue')
	# 	elif i=='I':color_map.append('red')
	# 	else:color_map.append('green') 
	# plt.figure(figsize=(12,8))	
	# nx.draw(G,pos=pos, node_color=color_map,node_size=40)
	# plt.pause(0.1)
	# plt.close()
plt.plot(np.arange(T),rhoS,label='rhoS',color='blue')
plt.plot(np.arange(T),rhoI,label='rhoI',color='red')
plt.plot(np.arange(T),rhoR,label='rhoR',color='green')
#plt.plot(np.arange(T),rhoR+rhoI+rhoS,label='rhoR',color='black')
#plt.plot(np.arange(1,T),R, label='R', color='black')
plt.legend()
plt.show()