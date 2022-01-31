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
T=100
p=0.01
Tequ=10
p_equ=1
theta=0.5
#
#

#Definition of the vector of object node and particle
class Particle(NamedTuple):
	index : int
	node : int
	state : str
class Node(NamedTuple):
    index: int
    particles: list
    state: str
    k: int 
    Npart: int
#
#
#
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
#
#
#
# Initialisation of the vector of object nodes and particule
ParticlesList=[]
Npart=np.zeros(N) # number if particle in each nodes
for i in range(P):
	node=random.randint(0,N-1) #pick up a random node for the particle
	if i==0 : 
		state='I'
	else :
		state='S'
	ParticlesList.append(Particle(i,node,state))
	Npart[node]+=1
#print(Npart)
NodesList=[]
for i in range(N):
	k_i=G.degree(i)
	NodesList.append(Node(i,[],'S',k_i,Npart[i]))
for i in range(P):
	NodesList[ParticlesList[i].node].particles.append(ParticlesList[i])
#print(NodesList)
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


########Loop only on deplacement to reach stationary state of node pop.
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
	print(t, end='\r')
# Move particles from  node i to node j with proba D[i,j]
	
	ParticlesStackALL=[]
	for i in range(N):
		ParticlesStack_i=[]
		for particle in NodesList[i].particles :
			j=random.choices(np.arange(N),D[i,:])[0]
			if i != j :
				ParticlesStack_i.append(particle)
				particle=particle._replace(node=j)
				ParticlesStackALL.append(particle)
				ParticlesList[particle.index]=ParticlesList[particle.index]._replace(node=j)
		for particle in ParticlesStack_i:
			NodesList[i].particles.remove(particle)			
	for particle in ParticlesStackALL:
			NodesList[particle.node].particles.append(particle)	
			
	#set colormap and size map for the figure
	color_map=[]
	size_map=[]
	for node in NodesList :
		N_S=0
		N_I=0
		N_R=0
		for particle in node.particles:
				if particle.state=='S':
					N_S+=1
				elif particle.state=='I':
					N_I+=1
				else:
					N_R+=1
		#print(N_S,N_I,N_R,N_S+N_I+N_R,Npart[node.index])
		if N_I==0:
			if N_S==0:node=node._replace(state='R')
			elif N_R==0: node=node._replace(state='S')
			else: node=node._replace(state='SR')
		else: node=node._replace(state='I')
	


		#set colormap for graph
		if node.state =='S':color_map.append('blue')
		elif node.state=='I':color_map.append('red')
		elif node.state=='SR': color_map.append('darkcyan')
		else:color_map.append('green')
		
		#set size map
		node=node._replace(Npart=len(node.particles))
		NodesList[node.index]=node
		size_map.append(node.Npart)		
	plt.figure(figsize=(12,8))	
	nx.draw(G,pos=pos, node_color=color_map,node_size=size_map)
	plt.pause(0.01)
	plt.close()

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
RhoPop_S=np.zeros(T)	
RhoPop_I=np.zeros(T)
RhoPop_R=np.zeros(T)
for t in range(0,T):
	print('t=',t, end='\r')
# Move particles from  node i to node j with proba A[i,j]
	ParticlesStackALL=[]
	for i in range(N):
		ParticlesStack_i=[]
		for particle in NodesList[i].particles :
			j=random.choices(np.arange(N),D[i,:])[0]
			if i != j :
				ParticlesStack_i.append(particle)
				particle=particle._replace(node=j)
				ParticlesStackALL.append(particle)
				ParticlesList[particle.index]=ParticlesList[particle.index]._replace(node=j)
		for particle in ParticlesStack_i:
			NodesList[i].particles.remove(particle)			
	for particle in ParticlesStackALL:
			NodesList[particle.node].particles.append(particle)		

# Inside each subpopulation node, the meanfield SIR dynamic.
	color_map=[]
	size_map=[]
	for node in NodesList:
		#set the right number of particle in the node
		node=node._replace(Npart=len(node.particles))

		N_I=0
		#print('before',node.index, node.state)
		for particle in node.particles :
			if particle.state=='I':
				N_I+=1
				if(N_I==1): node=node._replace(state='I')
		#print('after',node.index, node.state)
		#print('before',node.particles)
		if node.state=='I':
			ParticlesStack=[]
			for particle in node.particles:
				if particle.state=='S':
					statechoice=random.choices(['S','I'],[1-beta*N_I/node.Npart,beta*N_I/node.Npart])[0]
					#statechoice=random.choices(['S','I'],[1-beta,beta])[0]
					#statechoice=random.choices(['S','I'],[1-beta*1/node.Npart,beta*1/node.Npart])[0]
					particle=particle._replace(state=statechoice)
					ParticlesStack.append(particle)
					ParticlesList[particle.index]=ParticlesList[particle.index]._replace(state=statechoice)				
					
				elif particle.state=='I':
					statechoice=random.choices(['I','R'],[1-mu,mu])[0]
					#print(statechoice)
					particle=particle._replace(state=statechoice)
					ParticlesStack.append(particle)
					ParticlesList[particle.index]=ParticlesList[particle.index]._replace(state=statechoice)					
				elif particle.state=='R':
					ParticlesStack.append(particle)	
			node=node._replace(particles=ParticlesStack)
			NodesList[node.index]=node
			#print('Stack',ParticlesStack)
		#print('after',node.particles)	
		N_S=0
		N_I=0
		N_R=0
		for particle in node.particles:
				if particle.state=='S':
					N_S+=1
					RhoPop_S[t]=RhoPop_S[t]+iP
				elif particle.state=='I':
					N_I+=1
					RhoPop_I[t]+=iP
				else:
					N_R+=1
					RhoPop_R[t]+=iP	

		#print(N_S,N_I,N_R,N_S+N_I+N_R,Npart[node.index])
		if N_I==0:
			if N_S==0:node=node._replace(state='R')
			elif N_R==0: node=node._replace(state='S')
			else: node=node._replace(state='SR')
		else: node=node._replace(state='I')
	


		#set colormap for graph
		if node.state =='S':color_map.append('blue')
		elif node.state=='I':color_map.append('red')
		elif node.state=='SR': color_map.append('darkcyan')
		else:color_map.append('green')
		#set size map
		node=node._replace(Npart=len(node.particles))
		size_map.append(node.Npart)
	
	fig=(ax1, ax2) = plt.subplots(1, 2,figsize=(12,8))
	#fig=plt.figure(figsize=(12,8))	
	plt.subplot(211)	
	nx.draw(G,pos=pos, node_color=color_map,node_size=size_map)	
	plt.subplot(212)	
	plt.plot(np.arange(t),RhoPop_S[0:t],label='rhoS',color='blue')
	plt.plot(np.arange(t),RhoPop_I[0:t],label='rhoI',color='red')
	plt.plot(np.arange(t),RhoPop_R[0:t],label='rhoR',color='green')
	plt.legend()
	plt.pause(0.05)
	plt.close()	
plt.close()
plt.figure()	
plt.plot(np.arange(T),RhoPop_S,label='rhoS',color='blue')
plt.plot(np.arange(T),RhoPop_I,label='rhoI',color='red')
plt.plot(np.arange(T),RhoPop_R,label='rhoR',color='green')			
plt.legend()
plt.show()