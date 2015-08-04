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


#include <iostream>
#include <string>
#include <random>

#include <CL/cl.hpp>
#include "oclenv.h"
#include "customtypes.h"

ConfigData config = {
  100.0,  // output data size (MB)
  10.0,   // processing chunk size per enqueueNDRangeKernel call (MB)
  std::vector<uint32_t>() // specific gpus to use, if empty: use all available.
};

void CLArgs(int argc, char * argv[]);

int main(int argc, char * argv[])
{
  // Hanndle CLI parameters, if any

  CLArgs(argc, argv);

  // Set up the OpenCL environment and compile the kernels for each device

  OclEnv env;
  env.OclInit();

  env.OclDeviceInfo();

  env.NewCLCommandQueues();

  env.CreateKernels();

  printf("N_GPUs: %lu\n", config.gpu_select.size());

  // might seem superfluous but I'm validating the CLI input here against the
  // actual environment, so that the code doesn't try to use nonexistent
  // GPU #10000, etc
  env.SetGPUs(config.gpu_select);

  std::vector<uint32_t> gpus = env.GetGPUs();

  printf("OpenCL CommandQueues and Kernels ready.\n");

  // Set up I/O containers and fill input.

  if (static_cast<uint32_t>(config.data_size / config.chunk_size) < 1)
  {
    puts("GPU data size must be an integer multiple of the chunk size, \
      as padding is unsupported.");
    return 0;
  }

  float total_size = config.data_size * gpus.size();
  // total size of each input array, in MB
  float chunk_size = config.chunk_size;
  // size of each chunk summed by a single kernel execution

  uint32_t n = static_cast<uint32_t>(total_size * 1e6 / sizeof(float));
  uint32_t n_gpu = static_cast<uint32_t>(config.data_size * 1e6 /
    sizeof(float));
  uint32_t n_chunk = static_cast<uint32_t>( chunk_size * 1e6 / sizeof(float));
  uint32_t n_chunks = n_gpu / n_chunk;

  printf("Total Input Size: %.3f (MB), GPU Size: %.3f (MB), \
    Compute Chunk: %.3f (MB), Total Array Size: %d, GPU Array Size: %d\n",
      total_size, config.data_size, chunk_size, n, n_gpu);

  std::vector<float> input_one, input_two, output;

  input_one.resize(n);
  input_two.resize(n);
  output.resize(n);

  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0,1.0);

  puts("Generating random number sets...\n");
  for (uint32_t i = 0; i < n; i++)
  {
    input_one.at(i) = distribution(generator);
    input_two.at(i) = distribution(generator);
  }
  puts("Number sets complete.\n");

  uint32_t buffer_mem_size = n_chunk * sizeof(float);

  printf("N Chunks: %d, Chunk Buffer Size: %d (B)\n",
    n_chunks, buffer_mem_size);

  // OpenCL setup and kernel execution

  cl_int err;

  cl::Context * cntxt = env.GetContext();
  std::vector<cl::CommandQueue*> cqs;
  std::vector<cl::Kernel*> kerns;

  std::vector<cl::Buffer> ones;
  std::vector<cl::Buffer> twos;
  std::vector<cl::Buffer> outs;

  for (uint32_t d = 0; d < gpus.size(); d++)
  {
    cqs.push_back(env.GetCq(gpus.at(d)));
    kerns.push_back(env.GetKernel(gpus.at(d)));

    ones.push_back(cl::Buffer(  (*cntxt), // cl::Context &context
                                CL_MEM_READ_ONLY, // cl_mem_flags
                                buffer_mem_size, // size_t size
                                NULL, // void *host_ptr
                                &err // cl_int *err
                             ));
    if (CL_SUCCESS != err)
      env.Die(err);

    // Set up data container OpenCL buffers

    twos.push_back(cl::Buffer((*cntxt), CL_MEM_READ_ONLY, buffer_mem_size,
      NULL, &err));
    if (CL_SUCCESS != err)
      env.Die(err);
    outs.push_back(cl::Buffer((*cntxt), CL_MEM_WRITE_ONLY, buffer_mem_size,
      NULL, &err));
    if (CL_SUCCESS != err)
      env.Die(err);

    // Set the kernel arguments
    kerns.back()->setArg(0, ones.back());
    kerns.back()->setArg(1, twos.back());
    kerns.back()->setArg(2, outs.back());

  }

  cl::NDRange offset(0);
  cl::NDRange compute_range(n_chunk);

  // Execute the work sets

  for (uint32_t c = 0; c < n_chunks; c++)
  {
    // Write to the input buffers
    for (uint32_t d = 0; d < gpus.size(); d++)
    {
      err = cqs.at(d)->enqueueWriteBuffer(
        ones.at(d), // address of relevant cl::Buffer
        CL_FALSE, // non blocking
        static_cast<uint32_t>(0), // offset (bytes)
        buffer_mem_size, // total write size (bytes)
        &input_one.at(d * n_gpu + c * n_chunk), // pointer to root of data array
        NULL, // no events to wait on
        NULL // no events to link to for status updates
      );
      if (CL_SUCCESS != err)
        env.Die(err);

      err = cqs.at(d)->enqueueWriteBuffer(
        twos.at(d), // address of relevant cl::Buffer
        CL_FALSE, // non blocking
        static_cast<uint32_t>(0), // offset (bytes)
        buffer_mem_size, // total write size (bytes)
        &input_two.at(d * n_gpu + c * n_chunk), // pointer to root of data array
        NULL, // no events to wait on
        NULL // no events to link to for status updates
      );
      if (CL_SUCCESS != err)
        env.Die(err);

      cqs.at(d)->flush();
    }

    // execute the kernel
    for (uint32_t d = 0; d < gpus.size(); d++)
    {
      cqs.at(d)->finish();

      err = cqs.at(d)->enqueueNDRangeKernel(
        (*kerns.at(d)), // address of kernel
        offset, // starting global index
        compute_range, // ending global index
        cl::NullRange, // work items / work group (just 1)
        NULL,
        NULL
      );
      if (CL_SUCCESS != err)
        env.Die(err);

      cqs.at(d)->flush();
    }

    // read back the data
    for (uint32_t d = 0; d < gpus.size(); d++)
    {
      cqs.at(d)->finish();

      err = cqs.at(d)->enqueueReadBuffer(
        outs.at(d), // address of relevant cl::Buffer
        CL_FALSE, // execute and blocking
        static_cast<uint32_t>(0), // offset (bytes)
        buffer_mem_size, // total write size (bytes)
        &output.at(d * n_gpu + c * n_chunk), // pointer to root of data array
        NULL, // no events to wait on
        NULL  // no events to link to for status updates
      );

      if (CL_SUCCESS != err)
        env.Die(err);
    }

    // Don't need to clFinish here, we can start copying the new input data.
    // There's a blocking statement before each enqueueNDRangeKernel anyways.
  }

  // will clFinish here to make sure the last enqueueReadBuffers are done.
  for (uint32_t d = 0; d < gpus.size(); d++)
    cqs.at(d)->finish();

  printf("100.00%% complete\n");

  // random tests of correctness

  uint32_t n_tests = 20;

  printf("Testing %d random entries for correctness...\n", n_tests);

  std::uniform_int_distribution<uint32_t> int_distro(0, n);

  for (uint32_t i = 0; i < n_tests; i++)
  {
    uint32_t entry = int_distro(generator);

    printf("%.4f + %.4f = %.4f ? %.4f\n", input_one.at(entry),
      input_two.at(entry), output.at(entry),
        input_one.at(entry) + input_two.at(entry));
  }

  // cleanup

  return 0;
}

void CLArgs(int argc, char * argv[])
{
  std::vector<std::string> args(argv, argv+argc);
  bool set_datasize = false;
  bool set_chunksize = false;

  for (uint32_t i = 0; i < args.size(); i++)
  {
    if (args.at(i).find("-datasize") == 0)
    {
      config.data_size=std::stof(args.at(i).substr(args.at(i).find('=')+1));
      set_datasize = true;
    }
    else if (args.at(i).find("-chunksize") == 0)
    {
      config.chunk_size=std::stof(args.at(i).substr(args.at(i).find('=')+1));
      set_chunksize = true;
    }
    else if (args.at(i).find("-gpus") == 0)
    {
      std::string delim = ",";
      std::string begin = "=";
      size_t start = 0;
      size_t pos = 0;
      std::string source =
        args.at(i).erase(0, args.at(i).find("=")+begin.length());
      std::string temp;
      pos = source.find(delim, start);

      while ( pos != std::string::npos )
      {
        temp = source.substr(start, pos - start);
        start = pos + delim.length();
        config.gpu_select.push_back(std::stoul(temp));

        pos = source.find(delim, start);
      }

      temp = source.substr(start, pos - start);
      config.gpu_select.push_back(std::stoul(temp));
    }
  }

  if (set_datasize && !set_chunksize)
    config.chunk_size = config.data_size / 2;

  if (set_chunksize && !set_datasize)
    config.data_size = config.chunk_size * 2;
}

//EOF
