import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

# taking input matrix

def inp_m():
    rows = int(input("Enter number of rows: "))
    cols = int(input("Enter number of columns: "))
    print("Enter the elements row-wise:")
    matrix = []
    for i in range(rows):
        row = list(map(int, input().split()))
        matrix.append(row)
    array = np.array(matrix)
    print("Your matrix:")
    print(array)
    return array

#taking input vector

def inpv():
    v = np.array(list(map(float, input("enter the vector ").split())))
    return v

# defining matrix addition 

def ma(m1, m2):
    rm1, cm1 = m1.shape
    rm2, cm2 = m2.shape

    def ele(m,i,j):
        return m[i][j]
    rm = np.zeros((rm1, cm2))
    for i in range(rm1):
        for j in range(cm2):
            rm[i][j] = ele(m1,i,j) + ele(m2,i,j)
    return rm

# defining matrix multiplication

def mm(m1, m2):
    rm1,cm1 = m1.shape
    rm2,cm2 = m2.shape
    rm = np.zeros((rm1, cm2))
    def m1jc(m1, n):
        l = []
        for i in range(rm1):
            l.append(m1[i][n])
        return np.array(l)
    
    for i in range(rm1):
        for j in range(cm2):
            rm[i][j] = np.dot(m1[i], m1jc(m2,j))
    return rm

# is matrix orthogonal ?

def is_ortho(m):
    tm = m.T 
    rm = mm(m, tm)
    if np.linalg.det(rm) == 1:
        return True 
    return False 

# finding solutions to the system of linear equations for a matrix 

def ssle(m, n):
    det = np.linalg.det(m)
    if np.isclose(det, 0):
        print("Determinant of coefficient matrix is '0'")

    try:
        # Attempt to solve exactly
        x = np.linalg.solve(m, n)
        print("Exact solution found:")
        print(x)
        # Verify solution
        print("Check Ax == b:", np.allclose(mm(m, x), n))
        return x
    except np.linalg.LinAlgError:
        print("System is inconsistent or singular; finding least-squares solution.")
        x, residuals, rank, s = np.linalg.lstsq(m, n, rcond=None)
        print("Least-squares solution:")
        print(x)   
        return x

# find the linear transformation of a vector 'v', where transformation matrix is 'A'

def lintrans(A, v):
    return mm(A,v)

# find eigenvalues and eigenvectors of a matrix A 

def eig(A):
    evalues, evectors = np.linalg.eig(A)
    print("eigen values : ")
    for x in evalues:
        print(x)
    print("eigen vectors : ")
    for x in evectors:
        print(x)
    
    return evalues,evectors 

# finding linearly independent vectors 

def li(A):
    lind = []
    r = np.linalg.matrix_rank(A)
    nr, nc = A.shape 
    fullrank = False
    if r == nc:
        fullrank = True 
    def m1jc(m1, n):
        return m1[:,n] 
    if fullrank:
        for i in range(nc):
            lind.append(m1jc(A, i))
        return np.array(lind)
    listofcolumns = list(A[:,j] for j in range(nc))
    tm = np.array(listofcolumns[0])
    rtm = np.linalg.matrix_rank(tm)
    lind.append(listofcolumns[0])
    for i in range(1,nc):
        tm = np.vstack([tm, listofcolumns[i]])
        rtml = np.linalg.matrix_rank(tm)
        if rtml > rtm:
            lind.append(tm[-1])
            rtm = rtml 
    return lind

# projection of a vector onto another vector 

def projvu(v, u):
    vu = np.dot(v,u)
    uu = np.dot(u,u)
    if uu == 0:
        return np.zeros_like(v) 
    return np.array((vu/uu)*u)

# norm of a vector 

def norm(v):
    return np.linalg.norm(v)

# unit vector in direction of vector v 

def unitv(v):
    if norm(v) == 0:
        return v 
    return v/norm(v)

# gram-schmidit orthogonalisation, input the basis of a vector space, outputs orthogonal vectors and orthonormal vectors

def gso(m):
    nr, nc = m.shape
    orthov = []
    orthonv = []
    orthov.append(m[:,0])
    orthonv.append(unitv(orthov[0]))
    for i in range(1, nc):
        v = m[:,i]
        for j in range(len(orthonv)):
            tv = orthonv[j]*np.dot(orthonv[j],v)
            tvv = v - tv 
        
        orthov.append(tvv) 
        orthonv.append(unitv(tvv))
        print("orthogonal vectors : ")
        print(orthov)
        print("orthonormal vectors : ")
        print(orthonv)
    return np.array(orthov),np.array(orthonv)

# Linear Regression, input features X as matrix 'm', and results as vector 'y'

def linreg(m, v):
    reg = linear_model.LinearRegression()
    reg.fit(m, v)
    print("Coefficients:", reg.coef_)
    print("Intercept:", reg.intercept_)
    return reg.coef_, reg.intercept_





