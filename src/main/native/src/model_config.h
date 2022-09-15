#ifndef _MODEL_CONFIG_H
#define _MODEL_CONFIG_H


/// corresponds to GeometricType enum definition order
#define GEOMETRIC_TYPE_ID_FREE_POINT                 0
#define GEOMETRIC_TYPE_ID_LINE                       1
#define GEOMETRIC_TYPE_ID_FIX_LINE                   2
#define GEOMETRIC_TYPE_ID_CIRCLE                     3
#define GEOMETRIC_TYPE_ID_ARC                        4

/// corresponds to ConstraintType enum definition order
#define CONSTRAINT_TYPE_ID_FIX_POINT                  0
#define CONSTRAINT_TYPE_ID_PARAMETRIZED_XFIX          1
#define CONSTRAINT_TYPE_ID_PARAMETRIZED_YFIX          2
#define CONSTRAINT_TYPE_ID_CONNECT_2_POINTS           3
#define CONSTRAINT_TYPE_ID_HORIZONTAL_POINT           4
#define CONSTRAINT_TYPE_ID_VERTICAL_POINT             5
#define CONSTRAINT_TYPE_ID_LINES_PARALLELISM          6
#define CONSTRAINT_TYPE_ID_LINES_PERPENDICULAR        7
#define CONSTRAINT_TYPE_ID_EQUAL_LENGTH               8
#define CONSTRAINT_TYPE_ID_PARAMETRIZED_LENGTH        9
#define CONSTRAINT_TYPE_ID_TANGENCY                  10
#define CONSTRAINT_TYPE_ID_CIRCLE_TANGENCY           11
#define CONSTRAINT_TYPE_ID_DISTANCE_2_POINTS         12
#define CONSTRAINT_TYPE_ID_DISTANCE_POINT_LINE       13
#define CONSTRAINT_TYPE_ID_ANGLE_2_LINES             14
#define CONSTRAINT_TYPE_ID_SET_HORIZONTAL            15
#define CONSTRAINT_TYPE_ID_SET_VERTICAL              16

///
/// Sztywnosc sprezyny niska - glownie dla polaczenia pomiedzy punktem
/// zafiksowanym "{a,b}" i nie zafiksowanym "p*"
///
#define CONSTS_SPRING_STIFFNESS_LOW                  1.0           
#define CONSTS_SPRING_STIFFNESS_HIGH                29.0

///
/// Internal stiffness from p1 to p2 
///
#define  CIRCLE_SPRING_ALFA                         10.0


///
/// Configuration property
///
///
enum class ComputationMode {

    ///
    /// danse matrix mode
    ///
    ///
    DENSE_MODE = 1,

    ///
    /// sparse mode with matrix computed from coo format into csr
    ///
    SPARSE_MODE = 2,

    ///
    /// mixed mode that first round precompute Inverse Permuataion for next rounds for direct insertions.    
    ///
    /// "Compress Sparse Row Format"
    ///
    DIRECT_MODE = 3,

    /// 
    /// Persist all commands on COO journal as records in linear form [ `add or `set ].
    /// 
    /// Afterwards SoA vector is compacted into COO format.
    /// 
    COMPACT_MODE = 3,
};


/// 
/// Configuration property 
/// 
enum class SolverMode {

    /// 
    /// danse solver with LU 
    /// 
    DENSE_LU = 1,

    /// 
    /// default QR solver for sparse matrix
    /// 
    SPARSE_QR = 2,

    /// 
    /// Incomplete LU factorization solver for sparse matrix
    /// 
    SPARSE_ILU = 3,
};


///
/// Computation layout that will prepare matrix [ A ]  state  - this is stiffness matrix, and Jacobian matrix.
/// 
/// Set into computation context on each Iteration.
/// 
/// Derived variable from ( Computation Mode, Solver Mode )
///
enum ComputationLayout {

    ///
    /// Tensor is initalized as columnard dense vector with leading dimension
    ///
    ///
    DENSE_LAYOUT = 1,

    ///
    /// Tensor is inistalized in COO - "Coordinate Format"
    ///
    SPARSE_LAYOUT = 2,

    ///
    /// Tensor is computed in CSR direct form ( inverse permutation vector ;  "reordering" )
    ///
    /// "Compress Sparse Row Format"
    ///
    DIRECT_LAYOUT = 3,

    /// 
    /// Tensor is computed in COO format at execution of kernel as a journal of COO commands.
    /// 
    /// In post-processing, journal vector is sorted and reduced ( compaction ) into final COO form.
    /// 
    COMPACT_LAYOUT = 4
};



#endif // _MODEL_CONFIG_H