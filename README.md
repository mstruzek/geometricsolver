README
=============================================

Geometric Constraint Solver

Youtube Presentation:

http://www.youtube.com/watch?v=e1DlGjwGlkQ

=============================================


# Numerical model:

Static Equations of mechanical system  build from springs and unitary objects with singular mass (  primary equations ).

> F(q) - Q(q,p)* a    = 0

> Fi(q,p)             = 0


Where:

    q -                 - vector of free variables ( generalized constraints ), usually we will look and evaluate diffrences for those variables, so delta q =  dq = ^q 

    p                   - free parametrs vector descbing constant value constraines ( like const radius or const distance from line )

    F(q) = K*q          - vector forces beetween points

    K                   - stiffness matrix 

    Q (q,p)             - (d(Fi)/dq)' - Jacobian from Constraints Vector ,  ' - is Transponse symbol.

    a                   - Lagrangian coefficients - Lambda

    Fi(q , p )          - constraint vector , so these are constraint ( Perpendicular, Parallel, Distance, Tangency, ... other described ).


# Basic model ( 2 dimensional ) Geometric Primitives : 

    fixed point         - single visible point with fixed position on skatch plane.

                        [ a ] 1x2                   -> a fixed point 

    
    free point          - single point with unitary mass attached

                        [ a, b, p1] - 3x2               -> a fixed point, b fixed point, p1 moving point.

    line                - single visible line with 2 attached unitary mass.

                        [ a , b , p1 , p2 ] 4x2         ->  a fixed point, b fiexed point, p1 moving point, p2 moving point.

    circle              - single visible circle and guideline with 2 attached unitary mass.

                        [ a , b , p1 , p2]  4x2         ->  a fixed point, b fixed point , p1 center point , p2 radius point .
               
    
    
    arc                 - two visible guidelines for  arc with attached unitary mass.            

                        [ a , b , c , d , p1 , p2 , p3  ] - [a,b,c,d] fixed points , p1 center point , p2 first arm radius , p3 second arm radius.

    
    fixed line          - fixed line for coordinate system OX with OY

                        [ a ,  b ]                      -> a fiex point , b fixed point
    




# Numerical Equations


So we will apply Newton-Raphson method into those equations to linearize around point of equilibrium:

After solving with iterative methods for linear system ( like CGM, LU, ... ), system should converge into minimum energy with those applied and satisfied constraints.


First equations for stiffness and Lagrangian Forces applied from constraints:

> F(q) + Q(q,p)  * a    = 0

will be  

> dF/dq * ^q  + F(q)  +  d( Q')/dq * ^dq + Q(q) * a = 0         

( in second expression first form without  coefficient ).  Definition of Hessian  -  d( Q')/dq = d((dFi/dq)')/dq 

Second equation for constraints: 

> Fi(q,p)                       = 0

will be

> d(Fi)/dq * ^q + Fi(q,p)       = 0
    

# Constraint model

Usually constraint are described in terms of associated vectors or external parameters.

Simple equation for vectors are   L , K , M , N  where L = [ q1 , q2 ]'  K  = [ q3 , q4]'  M  = [ q5 , q6]  N = [ q7 , q8] 

Connected Points Constraint:
> Fi(q)  =   L - K        = 0   

Perpendicular Lines Constraint:
> Fi(q) = (L-K)' * ( M - N)  = 0            - scalar product   

Parallel Lines Constraint:
> Fi(q) = (L-K) x ( M - N)  = 0             - cross product 

 
All equations for  Jacobian's or Hessian's are derived from primary equations.


Above primary equations are rewritten into liner system equations applicable for GCM or LU solvers 

> A * x  = b

where x evaluates into ^q - delta-q - generalized coordinates. 

# Other Notes

- Date : 2022


- [ w strone macierzy do bezposredniego zapisu !]
- [ Solver - snapshoting , podarzanie za bledem ]

- [ JNI ] - otworzyc dzwignie do C++  -- jni.h  - handler do memory , read , write Byte , Integer , Long , Double, Float
- [ Selector ] Solver Selector - Local Host , GPU Blas H ( handmade ), GPU CGM , 
- [ Visitator ] -  zapis ByteBuffer free, [ Ax = b ] cudaHostFree, cudaHostMalloc(_) albo lokalnie albo z `cuda. 
- [  GPU ]  - sprawdzic model , zaciagnac bindowanie memory - caly model przenisc pod JNI i wiezy, stiffness.
- [ Constraint: CircleTangency ]
- 
- @ [!!!!! Error ] - Hessian Evaluation   -  iterable on keySet()  !!!! --   remove/add Primitives ( => Points ) - all constraints 3 wiezy !!!
- @ [ConstraintTangency, ConstraintLinesParallelism, ConstraintLinesPerpendicular ]
- @ [ Save ]   - przycisk :save model -> Writer : FORMAT PLIKU [ WSZYSTKIE OBIEKTY , NUMERY PUNKTOW , WIEZY , PARAMATRY]
- @ [ Load ]  -  przycisk :load model Reader
- @ [ Guides ] - show guidelines and  points  - gdzies zgubilsmy ta wersje z drukowaiem.
- @ [ auto KLMN  ]   - set K, L tuple when double clicked on K , or set auto K if db-clicked on L
- @ [ Relexed ] - przerobic Relaxed na random position for points - fluctuate - random shifts
- @ [ ConstrainVerticalPoints   : alignedOnX ]
- @ [ ConstrainHorizontalPoint : alignedOnY ]
- @ [ ConstraintDistancePointLine : extend Tangency ] - przygotowac rownania na kartce. !
- @ [ ConstraintParametrizedLength] - dlugoscie wzgledne nad wspolczynnik 
- @ [ ParametrizeXFix , ParametrizedYFix constraints]
- @ [ umiescmy formatke  Solver Space pod console logiem - (error, accTime, solvertTime, iter) ]
- 

## UNIFIED MATRIX INTERFACE
/*
* [  KERNEL Matrix Evaluations ] on-submit unit-of-work ->> model jest  zarzadzany  z  JNISolverGate
* ( Points >> , Primitives >> , Constraints >> )! .* 
* Gridem ( szykiem Kerneli ) -> uzupelanimy macierze A, b, x
* 1 - kernel - single primitive (8,8) 
* 1 - kernel - single jacobian tuple (1,2) , (2,2)
* 1 - kernel - single Hessian ( J,K ) GRID(32,32) == 1024  ~ iterate all constraints
* 
*/