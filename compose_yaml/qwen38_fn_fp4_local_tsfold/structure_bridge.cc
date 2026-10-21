// CPU-only Python/XGrammar bridge. Called only by explicitly structured requests.
#include <Python.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <dlfcn.h>
static std::recursive_mutex guard;
static PyObject *module = nullptr;
static bool initialize() {
    static bool attempted = false;
    if (attempted) return module != nullptr;
    attempted = true;
    if (!dlopen("libpython3.12.so.1.0", RTLD_NOW | RTLD_GLOBAL)) return false;
    const bool fresh = !Py_IsInitialized();
    PyGILState_STATE gil{};
    if (fresh) Py_Initialize(); else gil = PyGILState_Ensure();
    if (PyRun_SimpleString("import sys; sys.path.insert(0, '/opt/tf-structure')") == 0)
        module = PyImport_ImportModule("tf_structure");
    if (!module) PyErr_Print();
    if (fresh) PyEval_SaveThread(); else PyGILState_Release(gil);
    return module != nullptr;
}
extern "C" void *tf_structure_create(const char *model, const char *kind, const char *text) {
    std::lock_guard<std::recursive_mutex> lock(guard);
    if (!initialize()) return nullptr;
    auto gil = PyGILState_Ensure();
    PyObject *result = PyObject_CallMethod(module, "create", "sss", model, kind, text);
    if (!result) PyErr_Print();
    PyGILState_Release(gil);
    return result;
}
extern "C" int tf_structure_mask(void *state, const uint32_t *tokens, size_t count,
                                  uint32_t *output, size_t capacity) {
    std::lock_guard<std::recursive_mutex> lock(guard);
    auto gil = PyGILState_Ensure();
    PyObject *list = PyList_New(count);
    for (size_t i = 0; i < count; ++i) PyList_SET_ITEM(list, i, PyLong_FromUnsignedLong(tokens[i]));
    PyObject *result = PyObject_CallMethod(module, "mask", "OO", static_cast<PyObject *>(state), list);
    Py_DECREF(list);
    int words = -1;
    if (result && PyBytes_Check(result)) {
        auto n = PyBytes_Size(result);
        if (n > 0 && n % 4 == 0 && size_t(n) <= capacity * 4) {
            memcpy(output, PyBytes_AsString(result), n);
            words = int(n / 4);
        }
    }
    if (!result) PyErr_Print();
    Py_XDECREF(result);
    PyGILState_Release(gil);
    return words;
}
extern "C" void tf_structure_free(void *state) {
    std::lock_guard<std::recursive_mutex> lock(guard);
    auto gil = PyGILState_Ensure();
    Py_DECREF(static_cast<PyObject *>(state));
    PyGILState_Release(gil);
}
