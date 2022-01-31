import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import NamedTuple
#
#    
#Parameters:
N=100 #number of nodes
iN=1./N
P=100000  #number of particles
iP=1./P
beta=0.2 # proba S->I if I in node 
mu=0.01 # prob I->R # 
T=300
p=0.01
Tequ=100
p_equ=1
theta=0.5
#
#

class Node:
	def __init__(self, index, k):
		self.index = index
		self.k=k 
		self.Npart=0
		self.N_S=0
		self.N_I=0
		self.N_R=0
		self.state ='S'


# Weighted matrix defining mobility rates in between the nodes:
G = nx.generators.random_graphs.barabasi_albert_graph(N,1)
#G = nx.gnp_random_graph(N,4/N)
A=nx.adjacency_matrix(G)
W=np.empty((N,N))
D=np.empty((N,N))
#A=np.random.random((N, N))
#A=np.full((N, N), 0.5) 
#G = nx.gnp_random_graph(N,4./N)
#print(A.todense())
#nx.draw(G,node_size=40)
#plt.show()

NodesList=[]
for i in range(N):
	k_i=G.degree(i)
	NodesList.append(Node(i,k_i))
for particle in range(P):
	i=random.choices(range(N))[0]#choice of the node we attribute particle
	if particle==0:
		NodesList[i].N_I+=1
		NodesList[i].Npart+=1
		NodesList[i].state='I'
	else:
		NodesList[i].N_S+=1
		NodesList[i].Npart+=1
			
#for i in range(N):print(NodesList[i].index,NodesList[i].k, NodesList[i].Npart,NodesList[i].N_S,NodesList[i].N_I,NodesList[i].N_R,NodesList[i].state )
#
#
####Homogeneous diffusion (random walk)
# for i in range(0,N):
# 	k_i=G.degree(i)
# 	for j in range(0,N):
# 		D[i,j]=A[i,j]*p/k_i
# print(D)

####Traffic Dependent Mobility rates
# p=1
# w0=1
# for i in range(0,N):
# 	k_i=G.degree(i)
# 	T_i=0
# 	for j in range(0,N):
# 			k_j=G.degree(j)
# 			W[i,j]=A[i,j]*w_0*pow((k_i*k_j),theta)
# 			T_i+=W[i,j]
	
# 	for j in range(0,N):
# 		D[i,j]=p*W[i,j]/T_i			
#print(D)
#####Population Dependent Mobility rate
# for i in range(0,N):
# 	k_i=G.degree(i)
# 	T_i=0
# 	for j in range(0,N):
# 		k_j=G.degree(j)
# 		D[i,j]=A[i,j]*w_0*pow((k_i*k_j),theta)/float(Npart[i])
# print(D)			

#Set up the graph figure fixed		
seed = 1 
pos = nx.spring_layout(G, seed=seed)
plt.figure(figsize=(12,8))
#
#


#########Loop only on deplacement to reach stationary state of node pop.
for i in range(0,N):
	k_i=G.degree(i)
	T_i=0
	for j in range(0,N):
			k_j=G.degree(j)
			W[i,j]=A[i,j]*pow(float(k_j),theta)
			T_i+=W[i,j]	#traffic in i
	for j in range(0,N):
		D[i,j]=p_equ*W[i,j]/T_i
	D[i,i]=1-p_equ		
#print(D)

for t in range(0,Tequ):
	print(t,end="\r")
	# Move particles from  node i to node j with proba D[i,j]
	N_ij=np.zeros((N,N))
	N_ij_S=np.zeros((N,N))
	N_ij_I=np.zeros((N,N))
	N_ij_R=np.zeros((N,N))
	for i in range(0,N):
		P_S=int(NodesList[i].N_S)
		P_I=int(NodesList[i].N_I)
		P_R=int(NodesList[i].N_R)
		for particle in range(P_S):
			j=random.choices(np.arange(N),D[i,:])[0]
			if i != j :
 		 		N_ij[i,j]+=1
 		 		NodesList[i].Npart-=1
 		 		N_ij_S[i,j]+=1
 		 		NodesList[i].N_S-=1
		for particle in range(P_I):
 			j=random.choices(np.arange(N),D[i,:])[0]
 			if i != j :
 		 		N_ij[i,j]+=1
 		 		NodesList[i].Npart-=1
 		 		N_ij_I[i,j]+=1
 		 		NodesList[i].N_I-=1
		for particle in range(P_R):
			j=random.choices(np.arange(N),D[i,:])[0]
			if i != j :
 		 		N_ij[i,j]+=1
 		 		NodesList[i].Npart-=1
 		 		N_ij_R[i,j]+=1
 		 		NodesList[i].N_R-=1 		 		
 		 		
	for i in range(0,N):
		#print(NodesList[i].Npart)
		for j in range(0,N):
			NodesList[i].Npart+=N_ij[j,i]
			NodesList[i].N_S+=N_ij_S[j,i]
			NodesList[i].N_I+=N_ij_I[j,i]
			NodesList[i].N_R+=N_ij_R[j,i]

		if NodesList[i].N_I==0 :
			if NodesList[i].N_S==0 : NodesList[i].state='R'
			elif NodesList[i].N_R==0 : NodesList[i].state='S'
			else : NodesList[i].state='SR'
		else :
			NodesList[i].state='I'	 			


	# for i in range(N):
 # 		print(NodesList[i].index,NodesList[i].k, NodesList[i].Npart,NodesList[i].N_S,NodesList[i].N_I,NodesList[i].N_R,NodesList[i].state )

 		
		 			
	#set colormap and size map for the figure
# 	color_map=[]
# 	size_map=[]
# 	for node in NodesList :
# 		if node.state =='S': color_map.append('blue')
# 		elif node.state=='I' :color_map.append('red')
# 		elif node.state=='SR' : color_map.append('darkcyan')
# 		else : color_map.append('green')
# 		size_map.append(node.Npart)
# 	plt.figure(figsize=(12,8))	
# 	nx.draw(G,pos=pos, node_color=color_map,node_size=size_map)
# 	plt.pause(0.01)
# 	plt.close()
# print("")
############# setup for the epidemic porcess
####Traffic Dependent Mobility rates
for i in range(0,N):
	k_i=G.degree(i)
	T_i=0
	for j in range(0,N):
			k_j=G.degree(j)
			W[i,j]=A[i,j]*pow(float(k_j),theta)
			T_i+=W[i,j]	#traffic in i
	for j in range(0,N):
		D[i,j]=p*W[i,j]/T_i
	D[i,i]=1-p		
#print(D)


##Loop of time step
print('Epidemic!!')
Ntot_S=np.zeros(T)	
Ntot_I=np.zeros(T)
Ntot_R=np.zeros(T)
Ntot_I_new=np.zeros(T)
for t in range(0,T):
	print('t=',t,end="\r")
# Move particles from  node i to node j with proba A[i,j]
	N_ij=np.zeros((N,N))
	N_ij_S=np.zeros((N,N))
	N_ij_I=np.zeros((N,N))
	N_ij_R=np.zeros((N,N))
	for i in range(0,N):
		P_S=int(NodesList[i].N_S)
		P_I=int(NodesList[i].N_I)
		P_R=int(NodesList[i].N_R)
		for particle in range(P_S):
			j=random.choices(np.arange(N),D[i,:])[0]
			if i != j :
 		 		N_ij[i,j]+=1
 		 		NodesList[i].Npart-=1
 		 		N_ij_S[i,j]+=1
 		 		NodesList[i].N_S-=1
		for particle in range(P_I):
 			j=random.choices(np.arange(N),D[i,:])[0]
 			if i != j :
 		 		N_ij[i,j]+=1
 		 		NodesList[i].Npart-=1
 		 		N_ij_I[i,j]+=1
 		 		NodesList[i].N_I-=1
		for particle in range(P_R):
			j=random.choices(np.arange(N),D[i,:])[0]
			if i != j :
 		 		N_ij[i,j]+=1
 		 		NodesList[i].Npart-=1
 		 		N_ij_R[i,j]+=1
 		 		NodesList[i].N_R-=1 		 		
 		 		
	for i in range(0,N):
		#print(NodesList[i].Npart)
		for j in range(0,N):
			NodesList[i].Npart+=N_ij[j,i]
			NodesList[i].N_S+=N_ij_S[j,i]
			NodesList[i].N_I+=N_ij_I[j,i]
			NodesList[i].N_R+=N_ij_R[j,i]

		if NodesList[i].N_I==0 :
			if NodesList[i].N_S==0 : NodesList[i].state='R'
			elif NodesList[i].N_R==0 : NodesList[i].state='S'
			else : NodesList[i].state='SR'
		else :
			NodesList[i].state='I'	 			


 # Inside each subpopulation node, the meanfield SIR dynamic.
	color_map=[]
	size_map=[]
	for node in NodesList:
 		#set the right number of particle in the node	  
 		if node.state=='I':
 			N_I_new=0
 			N_R_new=0
 			beta_eff=beta*node.N_I/node.Npart
 			for i in range(int(node.N_S)):
 				statechoice=random.choices(['S','I'],[1-beta_eff,beta_eff])[0]
 				if statechoice=='I':
 					N_I_new+=1
 			for i in range(int(node.N_I)):
 				statechoice=random.choices(['I','R'],[1-mu,mu])[0]
 				if statechoice=='R':
 					N_R_new+=1	
 			node.N_S-=N_I_new
 			node.N_I+=N_I_new
 			node.N_I-=N_R_new
 			node.N_R+=N_R_new	
 			Ntot_I_new[t]+=N_I_new	
 		
 		Ntot_S[t]+=node.N_S
 		Ntot_I[t]+=node.N_I
 		Ntot_R[t]+=node.N_R

 		if node.N_I==0:
 			if node.N_S==0:
 				node.state='R'
 				color_map.append('green')#set colormap for graph
 			elif node.N_R==0: 
 				node.state='S'
 				color_map.append('blue')
 			else: 
 				node.state='SR'
 				color_map.append('darkcyan')
 		else: 
 			node.state='I'
 			color_map.append('red')		
 		size_map.append(node.Npart)
	
	fig=(ax1, ax2) = plt.subplots(1, 2,figsize=(12,8))
	#fig=plt.figure(figsize=(12,8))	
	plt.subplot(211)	
	nx.draw(G,pos=pos, node_color=color_map,node_size=size_map)	
	plt.subplot(212)	
	#plt.plot(np.arange(t),Ntot_S[0:t],label='N_S',color='blue')
	#plt.plot(np.arange(t),Ntot_I[0:t],label='N_I',color='red')
	#plt.plot(np.arange(t),Ntot_R[0:t],label='N_R',color='green')
	plt.plot(np.arange(t),Ntot_I_new[0:t],label='N_I_new',color='black')
	plt.legend()
	plt.pause(0.05)
	plt.close()	
# plt.close()
# plt.figure()	
# plt.plot(np.arange(T),RhoPop_S,label='rhoS',color='blue')
# plt.plot(np.arange(T),RhoPop_I,label='rhoI',color='red')
# plt.plot(np.arange(T),RhoPop_R,label='rhoR',color='green')			
# plt.legend()
# plt.show()