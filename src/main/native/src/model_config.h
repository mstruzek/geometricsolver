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
#define CONSTS_SPRING_STIFFNESS_LOW                   1.0           
#define CONSTS_SPRING_STIFFNESS_HIGH                 29.0

///
/// Internal stiffness from p1 to p2 
///
#define  CIRCLE_SPRING_ALFA                        10.0

#endif // _MODEL_CONFIG_H