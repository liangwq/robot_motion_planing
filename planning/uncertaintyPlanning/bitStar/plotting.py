import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

sys.path.append(os.curdir)
#importo librerie mie
import env

#---------- uso per plottare man mano che calcola -----------------
class Plotting:


    #costruttore
    def __init__(self, x_start, x_goal):
        self.xI, self.xG = x_start, x_goal
        self.env = env.Env()
        self.obs_bound = self.env.obs_boundary
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle


    #animazione --> plotto nodi visitati
    def animation(self, nodelist, path, name, animation=False):
        self.plot_grid(name)
        self.plot_visited(nodelist, animation)
        #plt.show()
        self.plot_path(path)


    #animazione --> specifico RRT-Connect
    def animation_connect(self, V1, V2, path, name):
        self.plot_grid(name)
        self.plot_visited_connect(V1, V2)
        self.plot_path(path)


    #grid
    def plot_grid(self, name):
        fig, ax = plt.subplots()

        for (ox, oy, w, h) in self.obs_rectangle:
            ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        for (ox, oy, r) in self.obs_circle:
            ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        plt.plot(self.xI[0], self.xI[1], "bs", linewidth=3)
        plt.plot(self.xG[0], self.xG[1], "gs", linewidth=3)

        plt.title(name)
        plt.axis("equal")


    #ciclo sul parent e plotto fino al corrente (introduco delay per visualizzare)
    @staticmethod
    def plot_visited(nodelist, animation):
        if animation:
            count = 0
            for node in nodelist:
                count += 1
                if node.parent:
                    plt.plot([node.parent.S[0], node.S[0]], [node.parent.S[1], node.S[1]], "-g")
                    plt.gcf().canvas.mpl_connect('key_release_event',
                                                 lambda event:
                                                 [exit(0) if event.key == 'escape' else None])
                    if count % 10 == 0:
                        plt.pause(0.0001)
        else:
            for node in nodelist:
                if node.parent:
                    plt.plot([node.parent.S[0], node.S[0]], [node.parent.S[1], node.S[1]], "-g")


    #ciclo sul parent e plotto fino al corrente (introduco delay per visualizzare) --> RRT-Connect --> 2 alberi
    @staticmethod
    def plot_visited_connect(V1, V2):
        len1, len2 = len(V1), len(V2)

        for k in range(max(len1, len2)):
            if k < len1:
                if V1[k].parent:
                    plt.plot([V1[k].S[0], V1[k].parent.S[0]], [V1[k].S[1], V1[k].parent.S[1]], "-g")
            if k < len2:
                if V2[k].parent:
                    plt.plot([V2[k].S[0], V2[k].parent.S[0]], [V2[k].S[1], V2[k].parent.S[1]], "-g")

            plt.gcf().canvas.mpl_connect('key_release_event',
                                          lambda event: [exit(0) if event.key == 'escape' else None])

            if k % 2 == 0:
                plt.pause(0.001)

        plt.pause(0.01)


    #plotto il path (delay)
    @staticmethod
    def plot_path(path):
        if len(path) != 0:
            plt.plot([x[0] for x in path], [x[1] for x in path], '-r', linewidth=2)
            plt.pause(0.01)
        plt.show()
