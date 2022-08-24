


/// LU FActorirization 

--- pelna macierz nie symetryczna hermitowska 



cusolverStatus_t
cusolverDnDgetrf_bufferSize(cusolverDnHandle_t handle,
                      int m,
                      int n,
                      double *A,
                      int lda,
                      int *Lwork );
					  
					  

--- faktoryzcacja !					  
					  
cusolverStatus_t
cusolverDnDgetrf(cusolverDnHandle_t handle,
           int m,
           int n,
           double *A,
           int lda,
           double *Workspace,
           int *devIpiv,
           int *devInfo );					  
		   
		   
----


cusolverStatus_t
cusolverDnDgetrs(cusolverDnHandle_t handle,
           cublasOperation_t trans,
           int n,
           int nrhs,
           const double *A,
           int lda,
           const int *devIpiv,
           double *B,
           int ldb,
           int *devInfo );		   
		   
	

multi solver LU

op(A) x X = B 	
		   
B  - nrhs = 1 jeden 'b'		   



#####################



Jacobian Set przywroci zapis jak na poczatku --> ROW-ORDER !!!

-- i poprawic Jacobiany :: constraintOffset  "rowOrder"

-- setVector -- zorientowana kolumnowo !!

-- setVectorT -- zorietnowana wierszowo !! prawidlowe podstawienie !!!

-- przygotowac Tensorm::fromDeviceMemTransposed(ec->A + (ec->dimension) * constraintOffset, ec->dimension, 1)  
	
	to oznacz iz setValue , setVector , setSubTensor() odkladamy transponowae wartosci w traansponowane wspolrzedne !

