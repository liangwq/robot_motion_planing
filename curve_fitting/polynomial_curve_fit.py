import numpy as np
import matplotlib.pyplot as plt

# degrees to radian converter
def deg_to_rad(T):
    return (T/180)*np.pi

qi = float(input('qi = ')) # initial position                0, for testing
qi = deg_to_rad(qi)

vi = float(input('vi = ')) # initial velocity                0, for testing     
vi = deg_to_rad(vi)

aci = float(input('aci = ')) # initial acceleration          0, for testing
aci = deg_to_rad(aci)

qf = float(input('qf = ')) # final position                  90, for testing
qf = deg_to_rad(qf)

vf = float(input('vf = ')) # final velocity                  50, for testing
vf = deg_to_rad(vf)

acf = float(input('acf = ')) # final acceleration            60, for testing
acf = deg_to_rad(acf)

ti = float(input('ti = ')) # initial time                   0, for testing
tf = float(input('tf = ')) # final time                     9, for testing

## Quintic Polynomial
## Solve the solution for q(t)=c0+c1*t+c2*t**2+c3*t**3+c4*t**4+c5*t**5

#这个对应的就是上面的AF(t)矩阵
M = [[1, ti, ti**2, ti**3, ti**4, ti**5],
     [0, 1, 2*ti, 3*ti**2, 4*ti**3, 5*ti**4],
     [0, 0, 2, 6*ti, 12*ti**2, 20*ti**3],
     [1, tf, tf**2, tf**3, tf**4, tf**5],
     [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
     [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]]  

M = np.matrix(M)   

b = [[qi], [vi], [aci], [qf], [vf], [acf]]

a = np.linalg.inv(M) * b  #这个就是上面的p参数向量

x = np.arange(ti,tf,0.05)

def qt(t,c0,c1,c2,c3,c4,c5):
    return c0 + c1*t + c2*t**2 + c3*t**3 + c4*t**4 + c5*t**5

y = qt(x,a[0,0],a[1,0],a[2,0],a[3,0],a[4,0],a[5,0])
print(a)

plt.figure()
plt.plot(x,y,'r',linestyle='-')
plt.text(1,1.5,'P(t)=p0+p1*t+p2*t**2+p3*t**3+p4*t**4+p5*t**5')
plt.grid(True)
plt.show()
