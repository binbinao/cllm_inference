# cllm_inference 构建与安装 Makefile
# 用法：
#   make            —— 编译 release 版可执行文件
#   make install    —— 安装到 $(PREFIX)/bin（默认 /usr/local）
#   make clean      —— 清理编译产物

CXX      ?= clang++
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

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all install clean
