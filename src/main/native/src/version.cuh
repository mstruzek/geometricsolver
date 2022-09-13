#ifndef _VERSION_H_
#define _VERSION_H_


#ifdef __NVCC__

// stringify property
#define STRING2(x) #x
#define STRING(x) STRING2(x)

#if (__cplusplus >= 201703)
#pragma message("Compilation Requirments standard C++ " STRING(__cplusplus))
#else
#error C++ compiler required with capabilities of c++17 at least.
#endif


/// c:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat

#endif // __NVCC__

#endif // _VERSION_H_