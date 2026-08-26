# Leone binary archive

This archive supports Linux x86_64. Windows and macOS packages are not
available in v0.1.

Run `bin/leone doctor -m <model.gguf>` before inference.

Install for one user with:

```sh
./install.sh
```

Set `PREFIX` to choose another destination. The default is `$HOME/.local`.
CUDA and cuBLAS must be available at runtime.
