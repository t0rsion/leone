# Leone binary archive

This archive supports Linux x86_64. Windows and macOS packages are not
available.

Before inference, run `bin/leone doctor -m <model.gguf>`.

Install for one user with:

```sh
./install.sh
```

Set `PREFIX` to choose another destination. The default is `$HOME/.local`.
CUDA and cuBLAS must be available at runtime.

The archive includes proof-gated SM89 plans under `plans/`. After installation,
they are under `$PREFIX/share/leone/plans`, with their search receipts under
`$PREFIX/share/leone/receipts`. A plan works only with its exact model SHA-256
and the tested compute capability.
