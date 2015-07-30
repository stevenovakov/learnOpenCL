# Makefile
#     for learnOpenCL
#     Copyright (C) 2015 Steve Novakov
#
#     This program is free software; you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation; either version 2 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License along
#     with this program; if not, write to the Free Software Foundation, Inc.,
#     51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# NOTE: MAKE SURE ALL INDENTS IN MAKEFILE ARE **TABS** AND NOT SPACES
#

CPLR=g++

# libOpenCL.so and OpenCL header locations in CUDA 7.0
# I personally just make sure that LIB_OPENCL is in system LD_LIBRARY_PATH
# and that /usr/local/cuda/include is in C_INCLUDE_PATH and CPLUS_INCLUDE_PATH,
# and then use system  env variables to compile.

LIBS= -lOpenCL

CPP_FLAGS = -Wall -ansi -pedantic -fPIC -std=c++11
DBG_FLAGS = -g -Wall -ansi -pedantic -fPIC -std=c++11
# This is for a bug with some GCC versions, involving std::thread/pthread where
# they don't get linked properly unless this gets thrown on the end of
# the compile statement
# see:
#  http://stackoverflow.com/questions/17274032/c-threads-stdsystem-error-operation-not-permitted?answertab=votes#tab-top
GCC_PTHREAD_BUG_FLAGS = -pthread -std=c++11

TARGET = program
OBJDIR = lib

SOURCES := $(wildcard *.cc)
OBJECTS := $(SOURCES:%.cc=$(OBJDIR)/%.o)

all: $(TARGET)

debug: CPP_FLAGS = $(DBG_FLAGS)
debug: $(TARGET)

# this results in a working program, but gdb doesn't see any debugging symbols
# even though the -g command is explicitly there
# $(TARGET): $(OBJDIR) $(OBJECTS)
# 	$(CPLR) $(CPP_FLAGS) -shared -o $(OBJDIR)/$(TARGET).so $(OBJECTS)
# 	$(CPLR) $(CPP_FLAGS) $(OBJDIR)/$(TARGET).so $(LD_FLAGS) -o $(TARGET) $(LIBS)

# gdb sees debugging symbols with this, but the executable segfaults before it
# makes it to the first line of main.cc. Think it's a problem with linking the
# libOpenCL.so  I have no idea why i would need to make a shared object first.

# cool, write about the differences between the driver based one in
# /usr/lib/x86_64-linux-gnu/ vs the cuda based one in /usr/local/cuda/lib64
# include the ln -ls, etc.

$(TARGET): $(OBJDIR) $(OBJECTS)
	$(CPLR) $(CPP_FLAGS) -o $(TARGET) $(OBJECTS) $(LIBS) $(GCC_PTHREAD_BUG_FLAGS)

$(OBJECTS): $(OBJDIR)/%.o:%.cc
	$(CPLR) $(CPP_FLAGS) -c $< -o $@ $(GCC_PTHREAD_BUG_FLAGS)

$(OBJDIR):
	@ mkdir -p $(OBJDIR)

clean:
	$(RM) $(TARGET) $(OBJECTS) $(OBJDIR)/$(TARGET).so
	$(RM) -rf $(OBJDIR)

g++ -Wall -ansi -pedantic -fPIC -std=c++11 -I/usr/local/cuda/include/ -L/usr/local/cuda/lib64/ -o program lib/main.o lib/oclenv.o -lOpenCL
