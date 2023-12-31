import math
import numpy as np
import os
import sys

sys.path.append(os.curdir)

import env

#definisco nodo
class Node:
    def __init__(self, S):
        self.S = np.array(S)
        self.Cov = np.array([[0.1, 0], [0, 0.1]]) #inizializzo covariance con Pw
        self.parent = None

#uso per tutti i planner
class Utils:

    #costruttore utils
    def __init__(self, p_safe = 0):
        self.env = env.Env()
        self.p_safe = p_safe
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

    #aggiorno ostacoli
    def update_obs(self, obs_cir, obs_bound, obs_rec):
        self.obs_circle = obs_cir
        self.obs_boundary = obs_bound
        self.obs_rectangle = obs_rec

    #get vertici ostacoli
    def get_obs_vertex(self, node):
        A = np.array([[-1, 0],[1, 0], [0, -1], [0, 1]])
        C = self.get_add_cost(node, A)
        obs_list = []
        for (ox, oy, w, h) in self.obs_rectangle:
            vertex_list = [[ox - C[0] , oy - C[2]],
                           [ox + w + C[1], oy - C[2]],
                           [ox + w + C[1] , oy + h + C[3]],
                           [ox - C[0] , oy + h + C[3]]]
            obs_list.append(vertex_list)
        return obs_list

    #controllo intersezioni in mezzo a X_near e X_new su cerchio
    def is_intersect_rec(self, start, end, o, d, a, b):
        v1 = [o[0] - a[0], o[1] - a[1]]
        v2 = [b[0] - a[0], b[1] - a[1]]
        v3 = [-d[1], d[0]] #rotazione positiva pi/2 di vettore X_start-->X_goal
        div = np.dot(v2, v3) #area tra v3 su v2
        if div == 0: #v3==v2 --> non penetra
            return False
        t1 = np.linalg.norm(np.cross(v2, v1)) / div #prodotto_vettoriale / prodotto_scalare
        t2 = np.dot(v1, v3) / div
        if t1 >= 0 and 0 <= t2 <= 1:
            shot = Node((o[0] + t1 * d[0], o[1] + t1 * d[1])) #mod
            dist_obs = self.get_dist(start, shot)
            dist_seg = self.get_dist(start, end)
            if dist_obs <= dist_seg:
                return True
        return False

    #controllo intersezioni in mezzo a X_near e X_new su cerchio
    def is_intersect_circle(self, o, d, a, r):
        d2 = np.dot(d, d)
        if d2 == 0:
            return False
        t = np.dot([a[0] - o[0], a[1] - o[1]], d) / d2
        if 0 <= t <= 1:
            shot = Node((o[0] + t * d[0], o[1] + t * d[1]))
            if self.get_dist(shot, Node(a)) <= r:
                return True
        return False

    #check collisioni tra X_near e X_new
    def is_collision(self, start, end):
        #controllo se X_start e X_goal sono in collisione
        if self.is_inside_obs(start) or self.is_inside_obs(end):
            return True
        #controllo se tra X_goal e X_start ci sono collisioni
        o, d = self.get_ray(start, end)
        obs_vertex = self.get_obs_vertex(start)
        for (v1, v2, v3, v4) in obs_vertex:
            if self.is_intersect_rec(start, end, o, d, v1, v2):
                return True
            if self.is_intersect_rec(start, end, o, d, v2, v3):
                return True
            if self.is_intersect_rec(start, end, o, d, v3, v4):
                return True
            if self.is_intersect_rec(start, end, o, d, v4, v1):
                return True
        for (x, y, r) in self.obs_circle:
            if self.is_intersect_circle(o, d, [x, y, 0], r):
                return True
        
        return False

    #controllo se X_near e X_new sono all'interno di Ostacoli
    def is_inside_obs(self, node):        
        #ostacoli cerchi
        A = np.array([[-1, 0],[1, 0], [0, -1], [0, 1]])
        C = np.mean(self.get_add_cost(node, A))
        for (x, y, r) in self.obs_circle:
            if math.hypot(node.S[0] - x, node.S[1] - y) <= r + C:
                return True
        #ostacoli rettangoli
        A = np.array([[-1, 0],[1, 0], [0, -1], [0, 1]])
        C = self.get_add_cost(node, A)
        for (x, y, w, h) in self.obs_rectangle:
            B = np.array([-x, x + w, -y, y + h])
            if  np.all(A @ node.S < B + C):
                return True
        #boundary esterno
        (xmin, xmax, ymin, ymax) = self.obs_boundary
        A = np.array([[1, 0],[-1, 0], [0, 1], [0, -1]])
        B = np.array([xmin, -xmax, ymin, -ymax])
        C = self.get_add_cost(node, A)
        if  np.all(A @ node.S < B + C):
            return True
        return False

    #costo aggiuntivo su constraint
    def get_add_cost(self, n, A):
        n_constraints = A.shape[0]
        Pv = np.zeros(n_constraints)
        for i in range(0, n_constraints):
            Pv[i] = np.sqrt(A[i] @ n.Cov @ np.transpose(A[i]))
        C = math.sqrt(2)*Pv*math.erf(1-self.p_safe)**-1
        return C

    #get direzione tra near e next
    @staticmethod
    def get_ray(start, end):
        orig = [start.S[0], start.S[1]]
        direc = [end.S[0] - start.S[0], end.S[1] - start.S[1]]
        return orig, direc

    #get distanza tra due punti
    @staticmethod
    def get_dist(start, end):
        return math.hypot(end.S[0] - start.S[0], end.S[1] - start.S[1])
