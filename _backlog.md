# Other Notes

- Date : 2022


- [ w strone macierzy do bezposredniego zapisu !]
- [ Solver - snapshoting , podarzanie za bledem ]

- [ JNI ] - otworzyc dzwignie do C++  -- jni.h  - handler do memory , read , write Byte , Integer , Long , Double, Float
- [ Selector ] Solver Selector - Local Host , GPU Blas H ( handmade ), GPU CGM ,
- [ Visitator ] -  zapis ByteBuffer free, [ Ax = b ] cudaHostFree, cudaHostMalloc(_) albo lokalnie albo z `cuda.
- [  GPU ]  - sprawdzic model , zaciagnac bindowanie memory - caly model przenisc pod JNI i wiezy, stiffness.
- [ Constraint: CircleTangency ]
- [ Help JOpitionPane -> TextArea ]

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