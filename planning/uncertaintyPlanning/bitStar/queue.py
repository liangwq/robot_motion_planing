import collections
import heapq

#firt in first out
class QueueFIFO:
    #costruttore
    def __init__(self):
        self.queue = collections.deque()
    #check empty
    def empty(self):
        return len(self.queue) == 0
    #aggiungo nuovo nodo (in fondo)
    def put(self, node):
        self.queue.append(node)
    #estraggo nodo da incima
    def get(self):
        return self.queue.popleft()

#last in last out
class QueueLIFO:
    #costruttore
    def __init__(self):
        self.queue = collections.deque()
    #check vuoto
    def empty(self):
        return len(self.queue) == 0
    #aggiungi in fondo
    def put(self, node):
        self.queue.append(node)  # enter from back
    #estrai da fondo
    def get(self):
        return self.queue.pop()  # leave from back

#riordina elementi usando valore di [priority]
class QueuePrior:
    #costruttore
    def __init__(self):
        self.queue = []
    #check empty
    def empty(self):
        return len(self.queue) == 0
    #aggiungo in pos=pos(priority[i])
    def put(self, item, priority):
        heapq.heappush(self.queue, (priority, item))
    #prendo da pos=pos(priority[i])
    def get(self):
        return heapq.heappop(self.queue)[1]
    #ritorno coda
    def enumerate(self):
        return self.queue
