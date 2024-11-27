Create debug build:

```bash
cmake -S . -B build_debug \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_FLAGS="-g -O0" \
  -DCMAKE_CXX_FLAGS="-g -O0"
```

Build:
```
cmake --build build_debug --config Debug -j10
```
