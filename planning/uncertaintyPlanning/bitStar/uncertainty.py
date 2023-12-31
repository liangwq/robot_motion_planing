import os
import sys
import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation as Rot
import gc

#inserisco directory corrente
sys.path.append(os.curdir)
#inserisco librerie mie
import env, plotting, utils, smoothing


#nodo standard (x, y, parent)
class Node:

    def __init__(self, S):
        self.S = np.array(S) #statostato
        self.Cov = [[0.1, 0], [0, 0.1]] #inizializzo covariance con Pw
        self.parent = None


#tree: [start, goal, V(vertex set), E(edge set), QE(coda edees), QV(coda vertex), Vold(salvo per confronto)]
class Tree:

    def __init__(self, x_start, x_goal):
        self.r = 4.0 #raggio di ricerca
        self.V = set() #set di vertici connessi
        self.E = set() #set di edges connessi
        self.QE = set() #coda ordinata di edges
        self.QV = set() #coda ordinata di vertici
        self.V_old = set() #set di vertici vecchi (li uso per confronto)


#BIT* algorithm
class BITStar:

    #costruttore
    def __init__(self, x_start, x_goal, eta, p_safe):
        
        #parametri iniziali
        self.x_start = Node(x_start)
        self.x_goal = Node(x_goal)
        self.eta = eta
        
        #definisco environment
        self.env = env.Env()
        self.utils = utils.Utils(p_safe)
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        #plotting
        
        self.plot_on = True
        self.animation_on = True
        if self.animation_on or self.plot_on:
            self.plotting = plotting.Plotting(x_start, x_goal)
            self.fig, self.ax = plt.subplots()

        #inizializzo albero con start e goal (lo uso sia per forward che reverse search)
        self.Tree = Tree(self.x_start, self.x_goal)

        #set e dict che uso
        self.X_unconn = set() #set di nodi non connessi ma potenzialmente migliarano soluzione
        self.g_F = dict() #dict dove salvo costo nodi
        
        #modello cinematico
        self.p_safe = p_safe #[0, 1] --> quanto voglio essere sicuro
        self.step_len = 0.5 #massima lunghezza singolo step e approssimazione
        self.dt = 0.1 #sampling time modello cinematico
        self.iter_max_local = 1000 #max numero simulazioni per raggiungere andare da nodo_A a nodo_B
        self.K = np.array([[1.9, 0], [0, 1.9]]) #guadagno per calcolare input: U = -K*dX
        self.A = [[1, 0], [0, 1]] #x(t+1) = A*x(t) + B*u(t) + Pw(t)
        self.B = [[self.dt,0], [0, self.dt]] #x(t+1) = A*x(t) + B*u(t) + Pw(t)
        self.Pw = np.array([[0.002, 0.001], [0.001, 0.002]]) #noise su modello cinematico

        #dati per benchmark
        self.iteration = 0
        self.max_total_time = 25
        self.benchmark_list = []


    #inizializzo nuovo batch
    def init(self):

        self.Tree.V.update([self.x_start, self.x_goal]) #aggiungo Xstart e X_goal a V(set di vertici
        self.X_unconn.add(self.x_goal) #aggiungo Xgoal a X_unconn(unconnected samples
        self.g_F[self.x_start] = 0.0 #costo vero Xstart = 0
        self.g_F[self.x_goal] = np.inf #costo vero Xgoal = inf

        #aggiorno parametri [cMin, theta, xCenter, C] che definiscono ovale di ricerca
        cMin, theta = self.calc_dist_and_angle(self.x_start, self.x_goal)
        C = self.RotationToWorldFrame(self.x_start, self.x_goal, cMin)
        xCenter = np.array([[(self.x_start.S[0] + self.x_goal.S[0]) / 2.0],
                            [(self.x_start.S[1] + self.x_goal.S[1]) / 2.0], [0.0]])
        m = 300 #numero di sample per batch

        return theta, cMin, xCenter, C , m, time.time()


    #planning
    def planning(self):

        theta, cMin, xCenter, C , m , start_time = self.init() #inizializzo parametri di ricerca (per primo batch)
        
        #numero di batch (numero massimo di volte che eseguo resampling)
        while (time.time() - start_time) < self.max_total_time:

            #controllo se QE e QV sono vuoti --> new batch
            if not self.Tree.QE and not self.Tree.QV:

                if self.plot_on:
                    if self.x_goal.parent is not None:
                        self.PlotFinalPath(True) #plotto path finale e interpolato con B-spline

                dm = self.Prune(self.g_F[self.x_goal]) #elimino tutti i vertici non inclusi nel nuovo ellisse
                self.X_unconn.update(self.Sample(m-0.*dm, self.g_F[self.x_goal], cMin, xCenter, C)) #resample nella nuovo regione selezionata con [m] samples
                self.Tree.V_old = {v for v in self.Tree.V} #salvo V << Vold per controllo successivo
                self.Tree.QV = {v for v in self.Tree.V} #salvo QV << V --> coda di vertici da processare in ordine di costo soluzione (vincolata a passare per nodi correnti in albero)


            #-------- costruisco tree con edge in ordine di costo -----------------------------------
            #BestVertexQueueValue(QV) --> stima miglior soluzione in QV --> espando solamente finche miglioramenti possibili
            while self.Tree.QV and self.BestVertexQueueValue() <= self.BestEdgeQueueValue():
                self.ExpandVertex(self.BestInVertexQueue()) #ogni vertice selezionato --> expando edges e aggiungo a QE (con condizioni)

            if not self.Tree.QE:
                break

            vm, xm = self.BestInEdgeQueue() #seleziono miglior edge in QE
            self.Tree.QE.remove((vm, xm)) #tolgo da QE il miglior edge (vm, xm) da processare
            #---------- check miglioramento(current_solution, current_tree) -------------------------
            if self.g_F[vm] + self.calc_dist(vm, xm) + self.h_estimated(xm) < self.g_F[self.x_goal]: #se stima ha potenziale --> procedo (risparmio collision check molte volte)
                actual_cost = self.cost(vm, xm) # <-- costo (vm, xm) che tiene conto di collisioni
                if self.g_estimated(vm) + actual_cost + self.h_estimated(xm) < self.g_F[self.x_goal]: #check con costo vero
                    if self.g_F[vm] + actual_cost < self.g_F[xm]: #controllo se anche cost_to_come(xm) migliorabile
                        
                        if xm in self.Tree.V: 
                            #xm era gia nell albero --> edge = wiring --> rimuovo edges(target_vertex, tree) 
                            edge_delete = set()
                            for v, x in self.Tree.E: #controllo gli edges che arrivano in xm nell albero
                                if x == xm: #aggiungo edge a lista eliminabili
                                    edge_delete.add((v, x))
                            for edge in edge_delete: #rimuovo edges inutili da albero
                                self.Tree.E.remove(edge)
                        else:
                            #xm non era nell albero --> edge = expansion --> sposto edges da X_unconn a QV (lo metto in coda per expansion)
                            self.X_unconn.remove(xm) #rimuovo xm da X_unconn --> non devo piu processarlo
                            self.Tree.V.add(xm) #aggiungo xm a albero
                            self.Tree.QV.add(xm) #aggiungo xm in coda a QV (non ho ancora assegnato costo)
                        
                        self.g_F[xm] = self.g_F[vm] + actual_cost #assegno a xm il suo costo nel dizionario
                        self.Tree.E.add((vm, xm)) #aggiungo edge (vm, xm) nell albero
                        xm.parent = vm #assegno vm come parent di xm
                        set_delete = set() #edges che non migliorano soluzione in QE
                        for v, x in self.Tree.QE:
                            #tutti gli edge (v, xm) con costo > nuovo cost_to_come(xm) --> aggiungo lista eliminabili
                            if x == xm and self.g_F[v] + self.calc_dist(v, xm) >= self.g_F[xm]:
                                set_delete.add((v, x))

                        #rimuovo edges eliminabili da QE
                        for edge in set_delete:
                            self.Tree.QE.remove(edge)
            else:
                #stima cost_to_come(x_goal) troppo alta --> svuoto e new batch (nessun altro edge puo migliorare soluzione)
                self.Tree.QE = set() #svuoto dict QE
                self.Tree.QV = set() #svuoto dict QV


            #salvo dati per benchmark
            self.iteration = self.iteration + 1

            #plotto
            if self.animation_on:
                if self.iteration % 5 == 0:
                    self.animation(xCenter, self.g_F[self.x_goal], cMin, theta)

        #plotto a fine batch
        if self.plot_on:
            if self.x_goal.parent is not None:
                self.PlotFinalPath(True) #plotto path finale e interpolato con B-spline 
            plt.show()

        self.benchmark_list.append([self.p_safe, self.g_F[self.x_goal]])
        print(self.g_F[self.x_goal])
        return self.benchmark_list


    #ciclo su parent --> estraggo x, y del path
    def ExtractPath(self):
        node = self.x_goal
        path_x, path_y = [node.S[0]], [node.S[1]]
        while node.parent:
            node = node.parent
            path_x.append(node.S[0])
            path_y.append(node.S[1])
        return path_x, path_y


    #prune --> aggiorno X_unconn, vertes_set, edge_set
    def Prune(self, cBest):
        #seleziono solo x in Xsample con stima costo minore di miglior costo reale corrente
        self.X_unconn = {x for x in self.X_unconn if self.f_estimated(x) < cBest}
        #seleziono solo vertici V nell albero con stima costo minore di miglior costo reale corrente
        self.Tree.V = {v for v in self.Tree.V if self.f_estimated(v) <= cBest} 
        #seleziono solo vertici edges (v, w) nell albero con stima costo (entrambi nodi) minore di miglior costo reale corrente
        self.Tree.E = {(v, w) for v, w in self.Tree.E
                       if self.f_estimated(v) <= cBest and self.f_estimated(w) <= cBest}
        #disconnetto da albero vertici con costo inf --> aggiungo a Xsample da riconnettere
        self.X_unconn.update({v for v in self.Tree.V if self.g_F[v] == np.inf})
        self.Tree.V = {v for v in self.Tree.V if self.g_F[v] < np.inf}

        return len(self.X_unconn)


    #ritorno costo come distanza tra start-end (inf se ostacolo)
    def cost(self, start, end):
        node_old = Node(start.S)
        count = 0 #limito numero massimo di simulazioni
        tot = 0 #costo totale accumulato
        while True:
            #seleziono input con feedback [U = K*(Xref-X)] --> input che mi spinge verso end
            U = np.dot(self.K, (end.S - node_old.S))
            #simulo in avanti --> node_new
            node_new = Node(np.dot(self.A, node_old.S) + np.dot(self.B, U))
            #aggiorno covariance node_new
            node_new.Cov = np.transpose(self.A) @ node_old.Cov @ self.A + self.Pw
            #distanza tra posizione corrente e node_end
            dist = self.calc_dist(node_old, end)
            #se ho collisione --> cost = inf
            if self.utils.is_collision(node_old, node_new):
                return np.inf    
            #se raggiungo target --> cost = tot (accumulato)
            if dist <= self.step_len:
                end.Cov = node_new.Cov #assegno covarianza accumulata a end
                tot += self.calc_dist(node_new, end) #assegno costo accumulato a end
                return tot
            #massimo numero simulazioni raggiunto --> no connessione --> cost = inf
            if count > self.iter_max_local:
                print("max local iter")
                return np.inf
            tot += self.calc_dist(node_old, node_new) #accumulo costo
            node_old = node_new
            count += 1


    #stima cost_to_come(x) + cost_to_go(x) --> costo path
    def f_estimated(self, node):
        return self.g_estimated(node) + self.h_estimated(node)


    #stima cost_to_come(x)
    def g_estimated(self, node):
        return self.calc_dist(self.x_start, node)


    #stima cost_to_go(x)
    def h_estimated(self, node):
        return self.calc_dist(node, self.x_goal)


    #creo m nuovi sample Xrand
    def Sample(self, m, cMax, cMin, xCenter, C):
        if cMax < np.inf: #se ho gia trovato un path --> stringo ellissoide
            return self.SampleEllipsoid(m, cMax, cMin, xCenter, C)
        else: #primo sample uniforme in c-space (posso subito applicare euristica volendo)
            return self.SampleFreeSpace(m)

    
    #plotto il path finale
    def PlotFinalPath(self, smooth):
        path_x, path_y = self.ExtractPath()
        if smooth: #se voglio interpolare con B-spline
            path_x, path_y = smoothing.approximate_b_spline_path(path_x, path_y, 4*len(path_x), degree=3)
        plt.plot(path_x, path_y, linewidth=2, color='r')
        plt.pause(0.01)


    #sample in regione definita da ellissoide
    def SampleEllipsoid(self, m, cMax, cMin, xCenter, C):
        r = [cMax / 2.0,
             math.sqrt(abs(cMax ** 2 - cMin ** 2)) / 2.0,
             math.sqrt(abs(cMax ** 2 - cMin ** 2)) / 2.0]
        L = np.diag(r)
        ind = 0
        Sample = set()
        #creo m sample
        while ind < m:
            xBall = self.SampleUnitNBall()
            x_rand = np.dot(np.dot(C, L), xBall) + xCenter
            node = Node((x_rand[(0, 0)], x_rand[(1, 0)])) #aggiungo covariance
            in_obs = self.utils.is_inside_obs(node) #check se sample interno a ostacolo
            in_x_range = self.x_range[0] <= node.S[0] <= self.x_range[1]
            in_y_range = self.y_range[0] <= node.S[1] <= self.y_range[1]
            #check se aggiungere o no
            if not in_obs and in_x_range and in_y_range:
                Sample.add(node)
                ind += 1
        return Sample


    #sampleFree normale
    def SampleFreeSpace(self, m):
        Sample = set()
        ind = 0
        while ind < m:
            node = Node((random.uniform(self.x_range[0], self.x_range[1]),
                        random.uniform(self.y_range[0], self.y_range[1])))
            if self.utils.is_inside_obs(node):
                continue
            else:
                Sample.add(node)
                ind += 1
        return Sample


    #aggiorno raggio di ricerca
    def radius(self, q):
        cBest = self.g_F[self.x_goal]
        lambda_X = len([1 for v in self.Tree.V if self.f_estimated(v) <= cBest])
        radius = 2 * self.eta * (1.5 * lambda_X / math.pi * math.log(q) / q) ** 0.5
        return radius


    #dato vertice V in QV --> creo edges che aggiungo a QE
    def ExpandVertex(self, v):
        self.Tree.QV.remove(v) #selexiono V e lo rimuovo da QV
        X_near = {x for x in self.X_unconn if self.calc_dist(x, v) <= self.Tree.r} #trovo set vicini di V nel set di sample non connessi
        #se potenzialmente soluzione migliorabile --> aggiungo edge
        for x in X_near:
            #g_estimated(v) = cost to come (v) , calc_dist(v, x) = calcolo costo da V a x , h_estimated(x) = stima cost to go(x) , g_F[self.x_goal] = costo soluzione corrente nell albero [inf se non ce]
            if self.g_estimated(v) + self.calc_dist(v, x) + self.h_estimated(x) < self.g_F[self.x_goal]:
                self.g_F[x] = np.inf #inizialmente setto costo a inf (correggo dopo)
                self.Tree.QE.add((v, x)) #aggiungo (v, x) in coda a QE
        #se V non era gia nell'albero trovo vicini di V nell albero corrente
        if v and v not in self.Tree.V_old:
            V_near = {w for w in self.Tree.V if self.calc_dist(w, v) <= self.Tree.r}
            for w in V_near:
                #se (v, w) not in E (edge non compreso nell albero) --> se puo migliorare soluzione [sia X_goal che w] --> inserisco (v, w) in QE
                if (v, w) not in self.Tree.E and \
                        self.g_estimated(v) + self.calc_dist(v, w) + self.h_estimated(w) < self.g_F[self.x_goal] and \
                        self.g_F[v] + self.calc_dist(v, w) < self.g_F[w]:
                    self.Tree.QE.add((v, w))
                    #se vertice w non ha costo assegnato --> cost = inf
                    if w not in self.g_F:
                        self.g_F[w] = np.inf


    #ritorno stima costo minore vertice
    def BestVertexQueueValue(self):
        if not self.Tree.QV:
            return np.inf
        return min(self.g_F[v] + self.h_estimated(v) for v in self.Tree.QV)


    #ritorno stima cost_to_come da V minore
    def BestEdgeQueueValue(self):
        if not self.Tree.QE:
            return np.inf
        return min(self.g_F[v] + self.calc_dist(v, x) + self.h_estimated(x)
                   for v, x in self.Tree.QE)


    #ritorno miglior vertice in coda QV con ordine corrente
    def BestInVertexQueue(self):
        if not self.Tree.QV:
            print("QV is Empty!")
            return None
        v_value = {v: self.g_F[v] + self.h_estimated(v) for v in self.Tree.QV}
        #leggo da dict QV key con val = min(cost)
        return min(v_value, key=v_value.get)


    #ritorno edge di QE con stima coso minore
    def BestInEdgeQueue(self):
        if not self.Tree.QE:
            print("QE is Empty!")
            return None, None
        e_value = {(v, x): self.g_F[v] + self.calc_dist(v, x) + self.h_estimated(x)
                   for v, x in self.Tree.QE}
        #seleziono vertice con costo minimo dal dizionario
        return min(e_value, key=e_value.get)


    #sample in ball raggio 1x1 (in 2D)
    @staticmethod
    def SampleUnitNBall():
        while True:
            x, y = random.uniform(-1, 1), random.uniform(-1, 1)
            if x ** 2 + y ** 2 < 1:
                return np.array([[x], [y], [0.0]])


    #trovo direzione x_start-x_goal (direzione) --> aggiorno C
    @staticmethod
    def RotationToWorldFrame(x_start, x_goal, L):
        a1 = np.array([[(x_goal.S[0] - x_start.S[0]) / L], [(x_goal.S[1] - x_start.S[1]) / L], [0.0]]) #vettore scalato x_start-->x_goal
        e1 = np.array([[1.0], [0.0], [0.0]])
        M = a1 @ e1.T #creo matrice 3x3 per applicare SVD
        U, _, V_T = np.linalg.svd(M, True, True) #SVD --> trovo direzioni principali per ruotare ellisse
        C = U @ np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(V_T.T)]) @ V_T
        return C


    #distanza tra due nodi
    @staticmethod
    def calc_dist(start, end):
        return math.hypot(start.S[0] - end.S[0], start.S[1] - end.S[1])


    #distanza e angolo tra due nodi
    @staticmethod
    def calc_dist_and_angle(node_start, node_end):
        dx = node_end.S[0] - node_start.S[0]
        dy = node_end.S[1] - node_start.S[1]
        return math.hypot(dx, dy), math.atan2(dy, dx)


    #definisco animation specifica di BIT* (non posso usare stessa di RRT...)
    def animation(self, xCenter, cMax, cMin, theta):
        plt.cla()
        self.plot_grid("Batch Informed Trees (BIT*)")
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        for v in self.X_unconn:
            plt.plot(v.S[0], v.S[1], marker='.', color='lightgrey', markersize='2')
        if cMax < np.inf:
            self.draw_ellipse(xCenter, cMax, cMin, theta)
        for v, w in self.Tree.E:
            plt.plot([v.S[0], w.S[0]], [v.S[1], w.S[1]], '-g')
        plt.pause(0.00001)


    #plotto tutto
    def plot_grid(self, name):
        for (ox, oy, w, h) in self.obs_rectangle:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )
        for (ox, oy, r) in self.obs_circle:
            self.ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )
        plt.plot(self.x_start.S[0], self.x_start.S[1], "rs", linewidth=3)
        plt.plot(self.x_goal.S[0], self.x_goal.S[1], "rs", linewidth=3)
        plt.title(name)
        plt.axis("equal")


    #draw ellipse specifico per BIT*
    @staticmethod
    def draw_ellipse(x_center, c_best, dist, theta):
        a = math.sqrt(abs(c_best ** 2 - dist ** 2)) / 2.0
        b = c_best / 2.0
        angle = math.pi / 2.0 - theta
        cx = x_center[0]
        cy = x_center[1]
        t = np.arange(0, 2 * math.pi + 0.1, 0.2)
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        rot = Rot.from_euler('z', -angle).as_matrix()[0:2, 0:2]
        fx = rot @ np.array([x, y])
        px = np.array(fx[0, :] + cx).flatten()
        py = np.array(fx[1, :] + cy).flatten()
        plt.plot(cx, cy, marker='.', color='darkorange')
        plt.plot(px, py, linestyle='--', color='darkorange', linewidth=2)


#=================== MAIN ==================================================
def main():
    x_start = (5, 5)
    x_goal = (46, 27)

    env = "BIT.npy"
    results = []
    iterations = 1
    
    for i in range(iterations):
        print("iteration = " + str(i))
        print("\n\n iterazione = " + str(i) + "\n\n")
        BIT = BITStar(x_start = x_start, x_goal = x_goal, eta = 2, p_safe = 0.8)
        results.append(BIT.planning()) #parto con planning

    #data_dir = os.path.join(os.curdir, os.path.join("data", os.path.join("results", env)))
    #np.save(data_dir, results)


if __name__ == '__main__':
    main()
