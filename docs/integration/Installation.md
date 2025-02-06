# Installing LibNTC into your project

Using CMake:
```cmake
set(LIBNTC_BIN_DIRECTORY "/path/to/your/build/output")
add_subdirectory(path/to/RTXNTC-Library)
target_link_libraries(<target_name> PRIVATE libntc)
```

Directly:
- Add `RTXNTC-Library/include` into the include paths - for the shaders too if implementing Inference on Sample
- Link `libntc.lib`
- Copy `libntc.dll` and `cudart64_12.dll` into the deployment location

Include the header:
```c++
#include <libntc/ntc.h>
```
