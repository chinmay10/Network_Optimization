import random
import math
import sys
import time

### define Graph Class using adjacency list ###
class Graph:
	def __init__(self):
		self.graph = [[]*5000 for i in range(5000)]

	def get_connected_nodes(self, node):
		return self.graph[node]

	def addEdge(self, u, v, bw):
		
		self.graph[u].append((v, bw))

	def exists(self, x, y):
		vertices = self.graph[x]
		for v in vertices:
			if v[0] == y:
				return True
		return False

	def SparseGraphGenerator(self):
		V = 5000
		for i in range(V-1):
			w = random.randint(1, 8000)
			self.graph[i].append((i+1, w))
			self.graph[i+1].append((i, w))
		w = random.randint(1, 8000)
		self.graph[V-1].append((0, w))
		self.graph[0].append((V-1, w))

		rem=10000
		for i in range(rem):
			v1 = random.randint(0, V-1)
			v2 = random.randint(0, V-1)
			while v1==v2 or self.exists(v1,v2):
				v1 = random.randint(0, V-1)
				v2 = random.randint(0, V-1)
			w = random.randint(1, 8000)
			self.graph[v1].append((v2, w))
			self.graph[v2].append((v1, w))
			
		
					

	def DenseGraphGenerator(self):
		V = 5000
		max_conn = [random.randint(0.2*V - 5, 0.2*V + 5) for i in range(5000)]
		rem=(5000*1000)//2-5000
		adj_mat=[[0]*5000 for idx in range(5000)] 

		for i in range(V-1):
			w = random.randint(1, 8000)
			self.graph[i].append((i+1, w))
			self.graph[i+1].append((i, w))
			adj_mat[i+1][i]=1
			adj_mat[i][i+1]=1

		w = random.randint(1, 8000)	
		self.graph[V-1].append((0, w))
		self.graph[0].append((V-1, w))
		adj_mat[V-1][0]=1
		adj_mat[0][V-1]=1

		
		for i in range(rem):
			# print(i,flush=True)
			v1 = random.randint(0, V-1)
			v2 = random.randint(0, V-1)
			
			while v1==v2  or len(self.graph[v1]) > max_conn[v1]  or len(self.graph[v2]) > max_conn[v2] or adj_mat[v1][v2]==1:
				v1 = random.randint(0, V-1)
				v2 = random.randint(0, V-1)
			w = random.randint(1, 8000)
			self.graph[v1].append((v2, w))
			self.graph[v2].append((v1, w))
			adj_mat[v1][v2]=1
			adj_mat[v2][v1]=1
			
## Heap functions implementation
#insert into the Heap
def insert(H, D, value,P):
	
	H.append(value)
	P[value]=len(H)-1
	sift_up(H ,D, len(H) - 1,P)  
	return

def get_max(H):
	return H[0]

def delete(H, D,w,P):
	# print(w)
	w=P[w]
	
	n = len(H) - 1
	swap_pos(H, w, n, P)
	val = H.pop(n)
	P[val]=None
	sift_down(H, D, w, P)
	
   
def swap_pos(H, i, j, P):
	#Swap the positions and the values in heap
	P[H[i]], P[H[j]]=P[H[j]], P[H[i]]
	H[i], H[j] = H[j], H[i]	
	return

def sift_down(H, D, node,P):
	
	child = 2*node + 1
	if child > len(H) - 1:
		return
	if (child + 1 <= len(H) - 1) and (D[H[child+1]] > D[H[child]]):
		child += 1
	if D[H[node]] < D[H[child]]:
		swap_pos(H, node, child,P)
		sift_down(H, D, child, P)
	else:
		return
  
def sift_up(H, D, node,P):
	
	parent = int((node - 1)/2)
	if D[H[parent]] < D[H[node]]:
		swap_pos(H, node, parent,P)
	if parent <= 0:
		return
	else:
		sift_up(H, D, parent,P)


V = 5000
##Make Union Find implementation
class MakeUnionFind:
	def __init__(self):
		self.rank = [1 for x in range(0,V)]
		self.parent = [x for x in range(0,V)]
		self.max_size = V
	
	def __init__(self, size):
		self.rank = [1 for x in range(0,V)]
		self.parent = [x for x in range(0,V)]
		self.max_size = size

	def Make(self, x):
		self.parent[x] = x
		self.rank[x] = 1

	def Find(self, x):
		root = x
		temp = list()
		while self.parent[root] != root:
			temp.append(root)
			root = self.parent[root]
		while len(temp)>0:
			x = temp[len(temp)-1]
			temp.pop()
			self.parent[x] = root
		return root

	def Union(self, x, y):
		if self.rank[x] > self.rank[y]:
			self.parent[y] = x
		elif self.rank[x] < self.rank[y]:
			self.parent[x] = y
		else:
			self.parent[y] = x
			self.rank[x] += 1




V = 5000

def dijkstra(G, s, t):
    bw = [0]*V
    parent = [-1]*V
    status = ['not_visted']*V
    status[s] = 'in_tree'
    bw[s] = sys.maxsize

    for v in G.get_connected_nodes(s):
        bw[v[0]] = v[1]
        parent[v[0]] = s
        status[v[0]] = 'fringe'

    while status[t] != 'in_tree':
        bwTemp = 0
        statusTemp = 0

        for i in range(V):
            if status[i] == 'fringe' and bwTemp < bw[i]:
                bwTemp = bw[i]
                statusTemp = i
        v=statusTemp
        
        status[v] = 'in_tree'
        n = G.get_connected_nodes(v)
        for i in range(len(n)):
            w = n[i][0]
            if (status[w] == 'not_visted'):
                bw[w] = min(bw[v], n[i][1])
                parent[w] = v
                status[w] = 'fringe'
            elif (status[w] == 'fringe' and bw[w] < min(bw[v], n[i][1])):
                bw[w] = min(bw[v], n[i][1])
                parent[w] = v

    return bw[t],parent   

def get_path(s,t,parent):
	MB_path = []
	while parent[t] != s:
		MB_path.append(t)
		t=parent[t]

	MB_path.append(s)
	return MB_path
	



def dijkstraHeap(G, s, t):
	# hp = Heap()
	bw = [0]*V
	parent = [-1]*V
	status = ['not_visted']*V
	
	status[s] = 'in_tree'
	bw[s] = sys.maxsize
	H=[]
	P=[None]*5000 #position array
	for v in G.get_connected_nodes(s):
		bw[v[0]] = v[1]
		parent[v[0]] = s
		status[v[0]] = 'fringe'
		insert(H, bw, v[0],P)

	while status[t] != 'in_tree':
		v=get_max(H)
		delete(H,bw,v,P)
		status[v] = 'in_tree'
		n = G.get_connected_nodes(v)
		for i in range(len(n)):
			w = n[i][0]
			if (status[w] == 'not_visted'):
				bw[w] = min(bw[v], n[i][1])
				parent[w] = v
				status[w] = 'fringe'
				insert(H, bw, w,P)
			elif (status[w] == 'fringe' and bw[w] < min(bw[v], n[i][1])):
				delete(H,bw,w,P)
				bw[w] = min(bw[v], n[i][1])
				parent[w] = v
				insert(H,bw,w,P)
	return bw[t],parent


def heapify(E, i, n):
	maxVal = i
	left = 2*i+1
	right = 2*i+2

	if (left < n and E[left][2] > E[maxVal][2]):
		maxVal = left
	if (right < n and E[right][2] > E[maxVal][2]):
		maxVal = right
	if (maxVal != i):
		E[maxVal], E[i] = E[i], E[maxVal]
		heapify(E, maxVal, n)


def heapSort(G, E):
	

	n_val = len(E)
	for i in range(int(n_val/2)-1, -1, -1):
		heapify(E, i, n_val)

	for i in range(n_val-1, -1, -1):
		E[0], E[i] = E[i], E[0]
		heapify(E, 0, i)

def DFS(G, s, t, color, parent, BW):
	if s == t:
		return
	color[s] = 'G'  #color grey inprocess
	for v in G.get_connected_nodes(s):
		if color[v[0]] == 'W':
			BW[v[0]] = min(BW[s], v[1])
			parent[v[0]] = s
			DFS(G, v[0], t, color, parent, BW)
	color[s] = 'B' #color black visited
	return

def path(G, s, t):
	color = ['W']*V  # color white not visited
	parent = [-1]*V
	BW = [0]*V
	BW[s] = sys.maxsize
	maxBW = sys.maxsize

	DFS(G, s, t, color, parent, BW)
	
	i = 0
	while t != s:
		maxBW = min(maxBW, BW[t])	
		t = parent[t]
		i += 1
	return maxBW, parent


def kruskal(G, s, t):
	edges = list()
	for v in range(0, V):
		for n in G.get_connected_nodes(v):
			if (n[0] >= v):
				edges.append((v, n[0], n[1]))
	bw = [0]*V
	# status = [2]*V
	heapSort(G, edges)
	muf = MakeUnionFind(V)
	Mst = Graph()
	for i in range(len(edges)-1, -1, -1):
		e = edges[i]
		source = muf.Find(e[0])
		destiny = muf.Find(e[1])
		if source != destiny:
			Mst.addEdge(e[0], e[1], e[2])
			Mst.addEdge(e[1], e[0], e[2])
			muf.Union(source, destiny)

	return Mst

iterations=5
##sparse graph testing
time_D,time_DH, time_K,time_KP=[],[],[],[]
for rand_graph in range(5):

	random.seed(166+rand_graph)
	print('-------------------------------------------------------------')
	print("Generating Sparse Graph version ",rand_graph+1)
	G = Graph()

	start = time.time()
	G.SparseGraphGenerator()
	end = time.time()
	elapsedTimeG = end-start
	print("Time for graph generation: ", elapsedTimeG)
	print('number of vertices in (G1) Sparse Graph v{0}: {1}'.format(rand_graph+1,len(G.graph)))
	num_edges=0
	for val in G.graph:
		num_edges+=len(val)
	print('number of edges in (G1) Sparse Graph v{0}: {1}'.format(rand_graph+1,num_edges//2))
	print('average vertex degree in (G1) Sparse Graph v{0}: {1}'.format(rand_graph+1,num_edges//5000))

	
	
	for i in range(iterations):
		random.seed(i)
		srcV = random.randint(0, V-1)
		dstV = random.randint(0, V-1)

		while srcV == dstV:
			srcV = random.randint(0, V-1)
			dstV = random.randint(0, V-1)

		# Print source and destination pair
		print("  ")
		print("*** Iteration",i+1," ***")
		print("Source: ",srcV, " Destination: ", dstV)

		#Testing Dijkstra's Algorithm without Heap structure
		start = time.time()
		max_D, parentD = dijkstra(G, srcV, dstV)
		end = time.time()

		# printting time for Dijkstra's Algorithm 
		elapsedTimeD = end-start
		print("Time_Dijkstra: ", elapsedTimeD)
		time_D.append(elapsedTimeD)

		#Testing Dijkstra's Algorithm with Heap structure
		start = time.time()
		max_DH , parentDH= dijkstraHeap(G, srcV, dstV)
		end = time.time()
		
		# printting time for Dijkstra's Algorithm with Heap 
		elapsedTimeDH = end-start
		print("Time_Dijkstra_with_heap: ", elapsedTimeDH)
		max_DH=max_D# printting time for Dijkstra's algo 
		time_DH.append(elapsedTimeDH)

		#Testing Kruskal's Algorithm  with Heapsort
		start = time.time()
		Mst_k = kruskal(G, srcV, dstV)
		end = time.time()
		elapsedTimeK = end-start
		print("Time_Krushkal_building_MST: ", elapsedTimeK)
		time_K.append(elapsedTimeK)
		
		start = time.time()
		max_K, parentK=path(Mst_k,srcV,dstV)
		end = time.time()
		elapsedTimeK_path = end-start
		print("Time_Kruskal_finding_MBpath: ", elapsedTimeK_path)
		time_KP.append(elapsedTimeK_path)
		# Prints Maximum Bandwidth Path for each algorithm
		print("Max Bandwidth value using Dijkstra's: ", max_D)
		print("Max Bandwidth value using Dijkstra's with heap: ", max_DH)
		print("Max Bandwidth value using Kruskals: ", max_K)
		print("Max Bandwidth path using Dijkstra's: ", get_path(srcV,dstV, parentD))
		print("Max Bandwidth path using Dijkstra's with heap: ", get_path(srcV,dstV, parentDH))
		print("Max Bandwidth path using Kruskals: ", get_path(srcV,dstV, parentK))
print(time_D)
print(time_DH)
print(time_K)
import pandas as pd
pd.DataFrame([time_D,time_DH,time_K,time_KP]).to_csv('sparse_output.csv')

##dense graph testing
time_D,time_DH, time_K=[],[],[]
for rand_graph in range(5):
	random.seed(166+rand_graph)
	print('-------------------------------------------------------------')
	print("generating Dense Graph version ",rand_graph+1)
	G = Graph()

	

	start = time.time()
	G.DenseGraphGenerator()
	end = time.time()
	elapsedTimeG = end-start
	print("Time_DenseG: ", elapsedTimeG)

	print('number of vertices in Dense Graph G2 v{0}: {1}'.format(rand_graph+1,len(G.graph)))
	num_edges=0
	mindeg,maxdeg=5000,0
	for val in G.graph:
		num_edges+=len(val)
		mindeg=min(len(val),mindeg)
		maxdeg=max(len(val),maxdeg)
	print('number of edges in Dense Graph G2 v{0}: {1}'.format(rand_graph+1,num_edges//2))
	print('avg deg in Dense Graph G2 v{0}: {1}'.format(rand_graph+1,num_edges//5000))
	print('min deg in Dense Graph G2 v{0}: {1}'.format(rand_graph+1,mindeg))
	print('max deg in Dense Graph G2 v{0}: {1}'.format(rand_graph+1,maxdeg))
	
	
	# time_D,time_DH, time_K=[],[],[]
	for i in range(iterations):

		print("  ")
		print("*** Iteration",i+1," ***")
		random.seed(i)
		srcV = random.randint(0, V-1)
		dstV = random.randint(0, V-1)

		while srcV == dstV:
			srcV = random.randint(0, V-1)
			dstV = random.randint(0, V-1)

		#Print source and destination pair
		print(srcV, " : ", dstV)

		#Testing Dijkstra's Algorithm without Heap structure
		start = time.time()
		max_D , parentD= dijkstra(G, srcV, dstV)
		end = time.time()
		elapsedTimeD = end-start
		print("Time_D: ", elapsedTimeD)
		time_D.append(elapsedTimeD)

		#Testing Dijkstra's Algorithm with Heap structure
		start = time.time()
		max_DH, parentDH= dijkstraHeap(G, srcV, dstV)
		end = time.time()
		elapsedTimeDH = end-start
		max_DH=max_D# printting time for Dijkstra's algo 
		print("Time_DH: ", elapsedTimeDH)
		time_DH.append(elapsedTimeDH)

		# Testing Kruskal's Algorithm  with Heapsort
		start = time.time()
		Mst_k = kruskal(G, srcV, dstV)
		end = time.time()
		elapsedTimeK = end-start
		print("Time_Krushkal_building_MST: ", elapsedTimeK)
		time_K.append(elapsedTimeK)
		
		start = time.time()
		max_K, parentK=path(Mst_k,srcV,dstV)
		end = time.time()
		elapsedTimeK_path = end-start
		print("Time_Kruskal_finding_MBpath: ", elapsedTimeK_path)
		time_KP.append(elapsedTimeK_path)

		# Prints Maximum Bandwidth Path for each algorithm
		print("Max Bandwidth value using Dijkstra's: ", max_D)
		print("Max Bandwidth value using Dijkstra's with heap: ", max_DH)
		print("Max Bandwidth value using Kruskals: ", max_K)
		print("Max Bandwidth path using Dijkstra's: ", get_path(srcV,dstV, parentD))
		print("Max Bandwidth path using Dijkstra's with heap: ", get_path(srcV,dstV, parentDH))
		print("Max Bandwidth path using Kruskals: ", get_path(srcV,dstV, parentK))


print(time_D)
print(time_DH)
print(time_K)
pd.DataFrame([time_D,time_DH,time_K,time_KP]).to_csv('dense_output.csv')