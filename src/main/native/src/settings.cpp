#include "settings.h"

#include <stdio.h>

namespace settings {

typedef union setting_value {
    bool boolValue;
    double doubleValue;
    long longValue;
} _value;

constexpr int SIZE = 64;
static setting_value _configuration[SIZE] = {};

bool get_configuration_value(int id, bool _value) { return _configuration[id].boolValue; }

double get_configuration_value(int id, double _value) { return _configuration[id].doubleValue; }

long get_configuration_value(int id, int _value) { return _configuration[id].longValue; }

void set_configuration_value(int id, bool value) { _configuration[id].boolValue = value; }

void set_configuration_value(int id, double value) { _configuration[id].doubleValue = value; }

void set_configuration_value(int id, int value) { _configuration[id].longValue = value; }

/**
 * Set settings from boolean property.
 * @param id
 * @param value
 * @return
 */
int setBooleanProperty(int id, bool value) {
    set_configuration_value(id, value);
    return 0;
}

/**
 * Set settings from long property.
 * @param id
 * @param value
 * @return
 */
int setLongProperty(int id, long value) {
    set_configuration_value(id, value);
    return 0;
}

/**
 * Set double settings property.
 * @param id
 * @param value
 * @return
 */
int setDoubleProperty(int id, double value) {
    set_configuration_value(id, value);
    return 0;
}

} // namespace settings