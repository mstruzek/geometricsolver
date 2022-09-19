#ifndef _SETTINGS_H_
#define _SETTINGS_H_


namespace settings {

bool get_configuration_value(int id, bool value);
double get_configuration_value(int id, double value);
long get_configuration_value(int id, int value);

void set_configuration_value(int id, bool value);
void set_configuration_value(int id, double value);
void set_configuration_value(int id, int value);

template <int Id, typename Type> struct ConfigurationKey {

    ConfigurationKey() {}

    ConfigurationKey(Type defaultValue) { reset(defaultValue); }

    Type value() { return get_configuration_value(Id, Type()); };

    explicit operator bool() const noexcept { return get_configuration_value(Id, Type()); }

    void reset(Type newValue) { set_configuration_value(Id, newValue); }

    static const int ID = Id;
};

static ConfigurationKey<1, bool> DEBUG_TENSOR_A = ConfigurationKey<1, bool>(false);
static ConfigurationKey<2, bool> DEBUG_TENSOR_B = ConfigurationKey<2, bool>(false);
static ConfigurationKey<3, bool> DEBUG = ConfigurationKey<3, bool>(false);
static ConfigurationKey<4, bool> DEBUG_TENSOR_SV = ConfigurationKey<4, bool>(false);

static ConfigurationKey<5, bool> CLOCK_MILLISECONDS = ConfigurationKey<5, bool>(true);
static ConfigurationKey<6, bool> CLOCK_NANOSECONDS = ConfigurationKey<6, bool>(true);

static ConfigurationKey<8, bool> DEBUG_SOLVER_CONVERGENCE = ConfigurationKey<8, bool>(false);
static ConfigurationKey<9, bool> DEBUG_CHECK_ARG = ConfigurationKey<9, bool>(false);

static ConfigurationKey<20, int> COMPUTATION_MODE = ConfigurationKey<20, int>(1);
static ConfigurationKey<21, int> SOLVER_MODE = ConfigurationKey<21, int>(1);
static ConfigurationKey<24, bool> SOLVER_INC_HESSIAN = ConfigurationKey<24, bool>(false);
static ConfigurationKey<26, double> SOLVER_LWORK_FACTOR = ConfigurationKey<26, double>(1.0);
static ConfigurationKey<27, double> SOLVER_EPSILON = ConfigurationKey<27, double>(10.0);

static ConfigurationKey<30, bool> DEBUG_CSR_FORMAT = ConfigurationKey<30, bool>(false);
static ConfigurationKey<31, bool> DEBUG_COO_FORMAT = ConfigurationKey<31, bool>(false);
static ConfigurationKey<60, bool> STREAM_CAPTURING = ConfigurationKey<60, bool>(false);

/**
 * Set boolean settings property.
 * DEBUG
 * DEBUG_MATRIX_A
 * DEBUG_TENSOR_B
 * DEBUG_TENSOR_SV
 * CLOCK_MILLISECONDS
 * CLOCK_NANOSECONDS
 * SOLVER_INC_HESSIAN
 * DEBUG_SOLVER_CONVERGENCE
 *
 * @param id
 * @param value
 * @return
 */

int setBooleanProperty(int id, bool value);

/**
 * Set long settings property.
 *
 * GRID_SIZE
 * BLOCK_SIZE
 *
 * @param id
 * @param value
 * @return
 */

int setLongProperty(int id, long value);

/**
 * Set double settings property.
 *
 * CU_SOLVER_LWORK_FACTOR
 *
 * @param id
 * @param value
 * @return
 */

int setDoubleProperty(int id, double value);

} // namespace settings


#endif // !_SETTINGS_H_