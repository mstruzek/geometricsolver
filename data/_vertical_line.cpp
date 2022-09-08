/// --signature: GeometricConstraintSolver  2009-2022
/// --file-format: V1
/// --file-name: e:\source\gsketcher\data\_vertical_line.cpp 

/// --definition-begin: 

err = jni_registerPointType(&env, eclass, 0, 1.655000e+02, 2.285000e+02); 
err = jni_registerPointType(&env, eclass, 1, 2.240000e+02, 2.420000e+02); 
err = jni_registerPointType(&env, eclass, 2, 3.410000e+02, 2.690000e+02); 
err = jni_registerPointType(&env, eclass, 3, 3.995000e+02, 2.825000e+02); 

err = jni_registerGeometricType(&env, eclass, 0, GEOMETRIC_TYPE_ID_LINE, 1, 2, -1, 0, 3, -1, -1); 


err = jni_registerConstraintType(&env, eclass, 0, CONSTRAINT_TYPE_ID_FIX_POINT, 0, -1, -1, -1, -1, 1.655000e+02, 2.285000e+02);
err = jni_registerConstraintType(&env, eclass, 1, CONSTRAINT_TYPE_ID_FIX_POINT, 3, -1, -1, -1, -1, 3.995000e+02, 2.825000e+02);
err = jni_registerConstraintType(&env, eclass, 2, CONSTRAINT_TYPE_ID_SET_VERTICAL, 1, 2, 0, 0, -1, 0.000000e+00, 0.000000e+00);


/// --definition-end: 

