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
  cl_int err;
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

  std::vector<cl::CommandQueue*> cqs;
  for (uint32_t d = 0; d < gpus.size(); d++)
  {
    cqs.push_back(env.GetCq(gpus.at(d)));
  }

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

  uint32_t buffer_mem_size = n_chunk * sizeof(float);
  uint32_t host_mem_size = n * sizeof(float);

  printf("N Chunks: %d, Chunk Buffer Size: %d (B)\n",
    n_chunks, buffer_mem_size);


  // std::vector<float> input_one, input_two, output;

  // input_one.resize(n);
  // input_two.resize(n);
  // output.resize(n);
  cl::Context * cntxt = env.GetContext();

  // pinned host memory
  cl::Buffer  one_host( (*cntxt),
                        CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                        host_mem_size, // size_t size
                        NULL, // void *host_ptr
                        &err // cl_int *err
                      );
  if (CL_SUCCESS != err)
    env.Die(err);

  cl::Buffer  two_host( (*cntxt),
                        CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                        host_mem_size, // size_t size
                        NULL, // void *host_ptr
                        &err // cl_int *err
                      );
  if (CL_SUCCESS != err)
    env.Die(err);

  cl::Buffer  out_host( (*cntxt),
                        CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                        host_mem_size, // size_t size
                        NULL, // void *host_ptr
                        &err // cl_int *err
                      );
  if (CL_SUCCESS != err)
    env.Die(err);

  // host-useable pointers to pinned host memory
  float * input_one, * input_two, *output;

  input_one = (float *) cqs.at(0)->enqueueMapBuffer(
    one_host,
    CL_TRUE,
    CL_MAP_WRITE,
    0,
    host_mem_size,
    NULL, NULL, NULL
  );

  input_two = (float *) cqs.at(0)->enqueueMapBuffer(
    two_host,
    CL_TRUE,
    CL_MAP_WRITE,
    0,
    host_mem_size,
    NULL, NULL, NULL
  );

  output = (float *) cqs.at(0)->enqueueMapBuffer(
    out_host,
    CL_TRUE,
    CL_MAP_READ,
    0,
    host_mem_size,
    NULL, NULL, NULL
  );

  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0,1.0);

  puts("Generating random number sets...\n");
  for (uint32_t i = 0; i < n; i++)
  {
    input_one[i] = distribution(generator);
    input_two[i] = distribution(generator);
  }
  puts("Number sets complete.\n");

  // OpenCL setup and kernel execution
  std::vector<cl::Kernel*> kerns;

  // device buffers
  std::vector<cl::Buffer> ones_dev;
  std::vector<cl::Buffer> twos_dev;
  std::vector<cl::Buffer> outs_dev;

  std::vector< std::vector<cl::Event> > kernel_events_1; // c = 0
  std::vector< std::vector<cl::Event> > kernel_events_2; // c != 0
  std::vector< std::vector<cl::Event> > read_events;

  // For every sub vector of events the order of events is:
  // 0) write one 1) write two 2) read_out 3) enqueueNDrangeKernel

  std::vector< std::vector<cl::Event> > * kernel_events = &kernel_events_1;

  for (uint32_t d = 0; d < gpus.size(); d++)
  {
    kernel_events_1.push_back(std::vector<cl::Event>());
    kernel_events_2.push_back(std::vector<cl::Event>());

    read_events.push_back(std::vector<cl::Event>());
    read_events.back().push_back(cl::Event());

    for (uint32_t i = 0; i < 2; i++)
      kernel_events_1.back().push_back(cl::Event());
    for (uint32_t i = 0; i < 3; i++)
      kernel_events_2.back().push_back(cl::Event());
  }


  for (uint32_t d = 0; d < gpus.size(); d++)
  {
    kerns.push_back(env.GetKernel(gpus.at(d)));

    ones_dev.push_back(cl::Buffer(  (*cntxt), // cl::Context &context
                                CL_MEM_READ_ONLY, // cl_mem_flags
                                buffer_mem_size, // size_t size
                                NULL, // void *host_ptr
                                &err // cl_int *err
                             ));
    if (CL_SUCCESS != err)
      env.Die(err);

    // Set up data container OpenCL buffers

    twos_dev.push_back(cl::Buffer((*cntxt), CL_MEM_READ_ONLY, buffer_mem_size,
      NULL, &err));
    if (CL_SUCCESS != err)
      env.Die(err);
    outs_dev.push_back(cl::Buffer((*cntxt), CL_MEM_WRITE_ONLY, buffer_mem_size,
      NULL, &err));
    if (CL_SUCCESS != err)
      env.Die(err);

    // Set the kernel arguments
    kerns.back()->setArg(0, ones_dev.back());
    kerns.back()->setArg(1, twos_dev.back());
    kerns.back()->setArg(2, outs_dev.back());

  }

  cl::NDRange offset(0);
  cl::NDRange compute_range(n_chunk);

  // Execute the work

  // For pinned memory:
  // need to map
  // do R/W
  // unmap
  // destroy

  for (uint32_t c = 0; c < n_chunks; c++)
  {
    // Write to the input buffers
    for (uint32_t d = 0; d < gpus.size(); d++)
    {
      err = cqs.at(d)->enqueueWriteBuffer(
        ones_dev.at(d), // address of relevant cl::Buffer
        CL_FALSE, // non blocking
        static_cast<uint32_t>(0), // offset (bytes)
        buffer_mem_size, // total write size (bytes)
        &input_one[d * n_gpu + c * n_chunk], // pointer to root of data array
        NULL, // no events to wait on
        &kernel_events->at(d).at(0) // output event info
      );
      if (CL_SUCCESS != err)
        env.Die(err);

      err = cqs.at(d)->enqueueWriteBuffer(
        twos_dev.at(d), // address of relevant cl::Buffer
        CL_FALSE, // non blocking
        static_cast<uint32_t>(0), // offset (bytes)
        buffer_mem_size, // total write size (bytes)
        &input_two[d * n_gpu + c * n_chunk], // pointer to root of data array
        NULL, // no events to wait on
        &kernel_events->at(d).at(1) // output event info
      );
      if (CL_SUCCESS != err)
        env.Die(err);

      cqs.at(d)->flush();
    }

    // execute the kernel
    for (uint32_t d = 0; d < gpus.size(); d++)
    {
      err = cqs.at(d)->enqueueNDRangeKernel(
        (*kerns.at(d)), // address of kernel
        offset, // starting global index
        compute_range, // ending global index
        cl::NullRange, // work items / work group (driver optimized)
        &kernel_events->at(d), // wait on these to be valid to execute
        &read_events.at(d).at(0) // output event info
      );

      if (CL_SUCCESS != err)
        env.Die(err);

      cqs.at(d)->flush();
    }

    // so that all subsequent enqueueNDRange calls wait on the read to finish
    if (c==0)
      kernel_events = &kernel_events_2;

    // read back the data
    for (uint32_t d = 0; d < gpus.size(); d++)
    {
      err = cqs.at(d)->enqueueReadBuffer(
        outs_dev.at(d), // address of relevant cl::Buffer
        CL_FALSE, // execute and blocking
        static_cast<uint32_t>(0), // offset (bytes)
        buffer_mem_size, // total write size (bytes)
        &output[d * n_gpu + c * n_chunk], // pointer to root of data array
        &read_events.at(d), // wait until kernel finishes to execute
        &kernel_events->at(d).at(2) // no events to link to for status updates
      );

      if (CL_SUCCESS != err)
        env.Die(err);

      cqs.at(d)->flush();
    }
  }

  // make sure the last reads are done
  for (uint32_t d = 0; d < gpus.size(); d++)
  {
    err = cl::Event::waitForEvents(kernel_events->at(d));
    if (CL_SUCCESS != err)
          env.Die(err);
  }

  printf("100.00%% complete\n");

  // random tests of correctness

  uint32_t n_tests = 20;

  printf("Testing %d random entries for correctness...\n", n_tests);

  std::uniform_int_distribution<uint32_t> int_distro(0, n);

  for (uint32_t i = 0; i < n_tests; i++)
  {
    uint32_t entry = int_distro(generator);

    printf("Entry %d -> %.4f + %.4f = %.4f ? %.4f\n", entry,
      input_one[entry], input_two[entry], output[entry],
        input_one[entry] + input_two[entry]);
  }

  // cleanup
  cqs.at(0)->enqueueUnmapMemObject( one_host,
                                    input_one,
                                    NULL,
                                    NULL
                                  );
  cqs.at(0)->enqueueUnmapMemObject( two_host,
                                    input_two,
                                    NULL,
                                    NULL
                                  );
  cqs.at(0)->enqueueUnmapMemObject( out_host,
                                    output,
                                    NULL,
                                    NULL
                                  );

  cqs.at(0)->finish();


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
