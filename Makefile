CC := g++ -std=c++0x

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	INCLUDES :=
	LIBS := -framework OpenCL
endif
ifeq ($(UNAME_S),Linux)
	INCLUDES := -I /usr/local/cuda-7.0/include/
	LIBS := -lOpenCL
endif

FLAGS := 

all: nbody

nbody: nbody.cpp
	$(CC) $(INCLUDES) $(FLAGS) -o $@ $< $(LIBS)

clean:
	rm nbody
