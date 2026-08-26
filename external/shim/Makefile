CC ?= cc
CFLAGS ?= -O2
CPPFLAGS += -I../llama.cpp/include -I../llama.cpp/ggml/include -I../llama.cpp/ggml/src
LDFLAGS += -L../llama.cpp/build/bin -Wl,-rpath,'$$ORIGIN/../../llama.cpp/build/bin'
LDLIBS += -lggml-base

.PHONY: all clean

all: build/ggml-dequant build/logits-dump

build/ggml-dequant: dequantize.c
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -std=c11 $< $(LDFLAGS) $(LDLIBS) -o $@

build/logits-dump: logits-dump.c ../llama.cpp/build/bin/libllama.so.0.2.0
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -std=c11 $< \
		../llama.cpp/build/bin/libllama.so.0.2.0 $(LDFLAGS) -o $@

clean:
	rm -f build/ggml-dequant build/logits-dump
