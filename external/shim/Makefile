CC ?= cc
CFLAGS ?= -O2
LLAMA_CPP_DIR ?= ../llama.cpp
LLAMA_CPP_BUILD_DIR ?= $(LLAMA_CPP_DIR)/build
CPPFLAGS += -I$(LLAMA_CPP_DIR)/include -I$(LLAMA_CPP_DIR)/ggml/include -I$(LLAMA_CPP_DIR)/ggml/src
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
LLAMA_BASE_LIBRARY := libggml-base.dylib
LLAMA_LIBRARY := libllama.dylib
LLAMA_CPP_RPATH ?= @loader_path/../../llama.cpp/build/bin
else
LLAMA_BASE_LIBRARY := libggml-base.so
LLAMA_LIBRARY := libllama.so.0.2.0
LLAMA_CPP_RPATH ?= $$ORIGIN/../../llama.cpp/build/bin
endif
LDFLAGS += -L$(LLAMA_CPP_BUILD_DIR)/bin
ifneq ($(strip $(LLAMA_CPP_RPATH)),)
LDFLAGS += -Wl,-rpath,'$(LLAMA_CPP_RPATH)'
endif
LDLIBS += -lggml-base

.PHONY: all clean

all: build/ggml-dequant build/logits-dump

build/ggml-dequant: dequantize.c $(LLAMA_CPP_BUILD_DIR)/bin/$(LLAMA_BASE_LIBRARY)
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -std=c11 $< $(LDFLAGS) $(LDLIBS) -o $@

build/logits-dump: logits-dump.c $(LLAMA_CPP_BUILD_DIR)/bin/$(LLAMA_LIBRARY)
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -std=c11 $< \
		$(LLAMA_CPP_BUILD_DIR)/bin/$(LLAMA_LIBRARY) $(LDFLAGS) -o $@

clean:
	rm -f build/ggml-dequant build/logits-dump
