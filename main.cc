/*
# main.cc
#     for learnOpenCL
#     Copyright (C) 2015 Steve Novakov

#     This program is free software; you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation; either version 2 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License along
#     with this program; if not, write to the Free Software Foundation, Inc.,
#     51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <random>

#include <CL/cl.hpp>
#include "oclenv.h"

int main(int argc, char **argv)
{
  // Set up the OpenCL environment and compile the kernels for each device

  OclEnv env;
  env.OclInit();

  env.OclDeviceInfo();

  env.NewCLCommandQueues();

  env.CreateKernels();

  printf("OpenCL CommandQueues and Kernels ready.\n");

  // Set up I/O containers and fill input.

  float total_size = 20.0;
  // total size of each input array, in MB
  float chunk_size = 2.0;
  // size of each chunk summed by a single kernel execution

  uint32_t n = static_cast<uint32_t>(total_size * 1e6 / 4.0);

  printf("Total Input Size: %.3f (MB), Compute Chunk: %.3f (MB), \
    Array Size: %d\n", total_size, chunk_size, n);

  std::vector<float> input_one, input_two, output;

  input_one.resize(n);
  input_two.resize(n);
  output.resize(n);

  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0,1.0);

  for(uint32_t i = 0; i < n; i++)
  {
    input_one.at(i) = distribution(generator);
    input_two.at(i) = distribution(generator);
  }

  uint32_t n_chunks = static_cast<uint32_t>(total_size / chunk_size);

  uint32_t n_chunk = static_cast<uint32_t>( chunk_size * n / total_size);
  uint32_t buffer_mem_size = n_chunk * sizeof(float);

  printf("N Chunks: %d, Chunk Buffer Size: %d (B)\n",
    n_chunks, buffer_mem_size);

  // OpenCL setup and kernel execution

  cl_int err;

  cl::Context * cntxt = env.GetContext();
  cl::CommandQueue * cq = env.GetCq(0); // just use the first device, for now
  cl::Kernel * kern = env.GetKernel(0); // just use the first device, for now

  // Set up data container OpenCL buffers

  cl::Buffer one_buffer, two_buffer, out_buffer;

  one_buffer = cl::Buffer(  (*cntxt), // cl::Context &context
                            CL_MEM_READ_ONLY, // cl_mem_flags
                            buffer_mem_size, // size_t size
                            NULL, // void *host_ptr
                            &err // cl_int *err
                        );

  if (CL_SUCCESS != err)
    env.Die(err);

  two_buffer = cl::Buffer( (*cntxt), CL_MEM_READ_ONLY, buffer_mem_size, NULL,
    &err);
  if (CL_SUCCESS != err)
    env.Die(err);
  out_buffer = cl::Buffer( (*cntxt), CL_MEM_READ_ONLY, buffer_mem_size, NULL,
    &err);
  if (CL_SUCCESS != err)
    env.Die(err);

  // Set the kernel arguments

  kern->setArg(0, one_buffer);
  kern->setArg(1, two_buffer);
  kern->setArg(2, out_buffer);

  cl::NDRange offset(0);
  cl::NDRange compute_range(n_chunk);

  // Execute the work sets

  for( uint32_t c = 0; c < n_chunks; c++)
  {
    // Write to the input buffers

    err = cq->enqueueWriteBuffer(
      one_buffer, // address of relevant cl::Buffer
      CL_FALSE, // non blocking
      static_cast<uint32_t>(0), // offset (bytes)
      buffer_mem_size, // total write size (bytes)
      &input_one.at(c * n_chunk), // pointer to root of data array
      NULL, // no events to wait on
      NULL // no events to link to for status updates
    );
    if (CL_SUCCESS != err)
      env.Die(err);

    err = cq->enqueueWriteBuffer(
      two_buffer, // address of relevant cl::Buffer
      CL_FALSE, // non blocking
      static_cast<uint32_t>(0), // offset (bytes)
      buffer_mem_size, // total write size (bytes)
      &input_two.at(c * n_chunk), // pointer to root of data array
      NULL, // no events to wait on
      NULL // no events to link to for status updates
    );
    if (CL_SUCCESS != err)
      env.Die(err);

    cq->finish(); // blocking : flush the write commands before proceeding

    // execute the kernel

    err = cq->enqueueNDRangeKernel(
      (*kern), // address of kernel
      offset, // starting global index
      compute_range, // ending global index
      cl::NullRange, // work items / work group (just 1)
      NULL,
      NULL
    );
    if (CL_SUCCESS != err)
      env.Die(err);

    cq->finish();

    err = cq->enqueueReadBuffer(
      out_buffer, // address of relevant cl::Buffer
      CL_TRUE, // execute and blocking
      static_cast<uint32_t>(0), // offset (bytes)
      buffer_mem_size, // total write size (bytes)
      &output.at(c * n_chunk), // pointer to root of data array
      NULL, // no events to wait on
      NULL  // no events to link to for status updates
    );
    if (CL_SUCCESS != err)
      env.Die(err);

    printf("%.2f%% complete\n",
     (100.0 * static_cast<float>(c) / static_cast<float>(n_chunks)));
    fflush(stdout);
  }
  printf("100.00%% complete\n");
  fflush(stdout);
  // random tests of correctness

  uint32_t n_tests = 10;

  printf("Testing 10 random entries for correctness...\n");

  std::uniform_int_distribution<uint32_t> int_distro(0, n);

  for (uint32_t i = 0; i < n_tests; i++)
  {
    uint32_t entry = int_distro(generator);

    printf("%.4f + %.4f = %.4f ?\n", input_one.at(entry),
      input_two.at(entry), output.at(entry));
  }

  // cleanup

  return 0;
}

//EOF
