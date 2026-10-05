# cllm_inference 构建与安装 Makefile
# 用法：
#   make            —— 编译 release 版可执行文件
#   make install    —— 安装到 $(PREFIX)/bin（默认 /usr/local）
#   make clean      —— 清理编译产物

# 自动检测 C++ 编译器：优先 clang++，其次 g++，最后回退 c++
# 注意：不能用 ?=，因为 make 内置变量 CXX 已有默认值（c++/g++），?= 不会覆盖
CXX := $(shell command -v clang++ 2>/dev/null || command -v g++ 2>/dev/null || echo c++)
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra
LDFLAGS  ?= -pthread
PREFIX   ?= /usr/local
DESTDIR  ?=

# 收集 src/ 下所有源文件（core + server）
SRCS := $(wildcard src/core/*.cpp src/server/*.cpp)
OBJS := $(SRCS:.cpp=.o)

TARGET := cllm_inference

all: $(TARGET)

$(TARGET): main.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) -Iinclude main.cpp $(OBJS) -o $@ $(LDFLAGS)

# 模式规则：编译 src/**/*.cpp 为目标文件
src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -Iinclude -c $< -o $@

install: $(TARGET)
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 0755 $(TARGET) $(DESTDIR)$(PREFIX)/bin/$(TARGET)

# 运行测试与基准
test:
	@sh tests/run_tests.sh

bench:
	@sh tests/run_tests.sh --bench

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all install test bench clean
