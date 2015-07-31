/*
# oclenv.cc
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
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

#include <CL/cl.hpp>

#include "oclenv.h"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
  static const std::string slash="\\";
#else
  static const std::string slash="/";
#endif

//*********************************************************************
//
// OclEnv Constructors/Destructors
//
//*********************************************************************
//
// Constructor(s)
//
OclEnv::OclEnv(){}

//
// Destructor
//
OclEnv::~OclEnv(){}

//*********************************************************************
//
// OclEnv Container Set/Get
//
//*********************************************************************

cl::Context * OclEnv::GetContext(uint32_t device_num)
{
  return &(this->ocl_contexts.at(device_num));
}

cl::CommandQueue * OclEnv::GetCq(unsigned int device_num)
{
  return &(this->ocl_device_queues.at(device_num));
}

cl::Kernel * OclEnv::GetKernel(unsigned int kernel_num)
{
  return &(this->kernel_set.at(kernel_num));
}

ConfigData * OclEnv::GetConfigData()
{
  return &(this->config_data);
}

void OclEnv::SetGPUs(std::vector<uint32_t> selected_gpus)
{
  if (this->ocl_devices.size() == 0)
  {
    puts("Can not assign gpus. Please initialize OpenCL Environment first.");
    return;
  }
  else
  {
    if (selected_gpus.size() == 0)
    {
      for (uint32_t g = 0; g< this->ocl_devices.size(); g++)
        this->desired_gpus.push_back(g);
    }
    else{
      for (uint32_t g = 0; g < this->ocl_devices.size(); g++)
      {
        if (std::find(selected_gpus.begin(), selected_gpus.end(), g)
          != selected_gpus.end() &&
          std::find(this->desired_gpus.begin(),this->desired_gpus.end(), g)
          == this->desired_gpus.end())
          this->desired_gpus.push_back(g);
      }
      std::sort(this->desired_gpus.begin(), this->desired_gpus.end());
    }
  }
}

std::vector<uint32_t> OclEnv::GetGPUs()
{
  return this->desired_gpus;
}

//*********************************************************************
//
// OclEnv OpenCL Interface
//
//*********************************************************************

//
// OclInit()
//
// Currently ignores all non-GPU devices

void OclEnv::OclInit()
{
  cl::Platform::get(&(this->ocl_platforms));

  if (0 == this->ocl_platforms.size())
  {
    printf("No OpenCL platforms found.\n");
    exit(-1);
  }

  cl_context_properties con_prop[3] =
  {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties) (this->ocl_platforms[0]) (),
    0
  };

  this->ocl_platforms.at(0);

  this->ocl_contexts.push_back(cl::Context(CL_DEVICE_TYPE_GPU, con_prop));

  this->ocl_devices.push_back(
    this->ocl_contexts.back().getInfo<CL_CONTEXT_DEVICES>().at(0));

  // 1 context per device

  for (uint32_t d = 1;
    d < this->ocl_contexts.back().getInfo<CL_CONTEXT_DEVICES>().size(); d++)
  {
    this->ocl_contexts.push_back(cl::Context(CL_DEVICE_TYPE_GPU, con_prop));
    this->ocl_devices.push_back(
      this->ocl_contexts.back().getInfo<CL_CONTEXT_DEVICES>().at(d));
  }

  printf("OpenCL Environment Initialized.\n");
}

void OclEnv::OclDeviceInfo()
{
  std::cout<<"\nLocal OpenCL Devices:\n";

  size_t siT[3];
  cl_uint print_int;
  cl_ulong print_ulong;
  std::string print_string;

  std::string device_name;

  for (std::vector<cl::Device>::iterator dit = this->ocl_devices.begin();
    dit != this->ocl_devices.end(); ++dit)
  {

    dit->getInfo(CL_DEVICE_NAME, &print_string);
    dit->getInfo(CL_DEVICE_NAME, &device_name);
    dit->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &print_int);
    dit->getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &siT);
    dit->getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &siT);

    std::cout<<"\tDEVICE\n";
    std::cout<<"\tDevice Name: " << print_string << "\n";
    std::cout<<"\tMax Compute Units: " << print_int << "\n";
    std::cout<<"\tMax Work Group Size (x*y*z): " << siT[0] << "\n";
    std::cout<<"\tMax Work Item Sizes (x, y, z): " << siT[0] <<
      ", " << siT[1] << ", " << siT[2] << "\n";

    dit->getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &print_ulong);
    std::cout<<"\tMax Mem Alloc Size: " << print_ulong << "\n";

    std::cout<<"\n";
  }
}

unsigned int OclEnv::HowManyDevices()
{
  return this->ocl_devices.size();
}

unsigned int OclEnv::HowManyCQ()
{
  return this->ocl_device_queues.size();
}

size_t OclEnv::GetKernelWorkGroupInfo(uint32_t device)
{
  size_t wg_size;
  this->kernel_set.at(device).getWorkGroupInfo<size_t>(
    this->ocl_devices.at(device), CL_KERNEL_WORK_GROUP_SIZE, &wg_size);
  return wg_size;
}

void OclEnv::NewCLCommandQueues()
{
  this->ocl_device_queues.clear();

  for (uint32_t k = 0; k < this->ocl_devices.size(); k++)
  {
    std::cout<<"Create CommQueue, Device: "<<k<<"\n";

    this->ocl_device_queues.push_back(
      cl::CommandQueue(this->ocl_contexts.at(k), this->ocl_devices.at(k),
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ||  CL_QUEUE_PROFILING_ENABLE));
  }
}

void OclEnv::CreateKernels()
{
  this->kernel_set.clear();

  cl_int err;

  // Read Source
  std::string fold = "kernels";

  std::string kernel_source = fold + slash + "summer.cl";
  std::string define_list =  "-I ./oclkernels";

  std::ifstream k_stream(kernel_source);
  std::string k_code(  (std::istreambuf_iterator<char>(k_stream) ),
                            (std::istreambuf_iterator<char>()));
  //
  // Build Program files here
  //
  // CAREFUL : this code assumes every device is identical
  //

  cl::Program::Sources k_source(
    1, std::make_pair(k_code.c_str(), k_code.length()));

  for (uint32_t d = 0; d < this->ocl_devices.size(); d++)
  {
    cl::Program k_program(cl::Program(this->ocl_contexts.at(d), k_source));

    err = k_program.build(this->ocl_devices);

    if (err != CL_SUCCESS)
    {
      std::cout<<"ERROR: " <<
        " ( " << this->OclErrorStrings(err) << ")\n";

      std::vector<cl::Device>::iterator dit = this->ocl_devices.begin();

      std::cout<<"BUILD OPTIONS: \n" <<
        k_program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(*dit) <<
         "\n";
      std::cout<<"BUILD LOG: \n" <<
        k_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*dit) <<"\n";

      exit(EXIT_FAILURE);
    }

  //
  // Compile Kernels from Program
  //

    this->kernel_set.push_back(
      cl::Kernel(k_program, "Summer", NULL));
  }
}

void OclEnv::Die(uint32_t reason, std::string additional)
{
  std::string error = this->OclErrorStrings(reason);
  puts(error.c_str());
  puts(additional.c_str());
  abort();
}

//
// Matches OCL error codes to their meaning.
//
// TODO: host in separate file and make accessible to entire software
//
std::string OclEnv::OclErrorStrings(cl_int error)
{
  const std::string cl_error_string[] =
  {
    "CL_SUCCESS",
    "CL_DEVICE_NOT_FOUND",
    "CL_DEVICE_NOT_AVAILABLE",
    "CL_COMPILER_NOT_AVAILABLE",
    "CL_MEM_OBJECT_ALLOCATION_FAILURE",
    "CL_OUT_OF_RESOURCES",
    "CL_OUT_OF_HOST_MEMORY",
    "CL_PROFILING_INFO_NOT_AVAILABLE",
    "CL_MEM_COPY_OVERLAP",
    "CL_IMAGE_FORMAT_MISMATCH",
    "CL_IMAGE_FORMAT_NOT_SUPPORTED",
    "CL_BUILD_PROGRAM_FAILURE",
    "CL_MAP_FAILURE",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "CL_INVALID_VALUE",
    "CL_INVALID_DEVICE_TYPE",
    "CL_INVALID_PLATFORM",
    "CL_INVALID_DEVICE",
    "CL_INVALID_CONTEXT",
    "CL_INVALID_QUEUE_PROPERTIES",
    "CL_INVALID_COMMAND_QUEUE",
    "CL_INVALID_HOST_PTR",
    "CL_INVALID_MEM_OBJECT",
    "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
    "CL_INVALID_IMAGE_SIZE",
    "CL_INVALID_SAMPLER",
    "CL_INVALID_BINARY",
    "CL_INVALID_BUILD_OPTIONS",
    "CL_INVALID_PROGRAM",
    "CL_INVALID_PROGRAM_EXECUTABLE",
    "CL_INVALID_KERNEL_NAME",
    "CL_INVALID_KERNEL_DEFINITION",
    "CL_INVALID_KERNEL",
    "CL_INVALID_ARG_INDEX",
    "CL_INVALID_ARG_VALUE",
    "CL_INVALID_ARG_SIZE",
    "CL_INVALID_KERNEL_ARGS",
    "CL_INVALID_WORK_DIMENSION",
    "CL_INVALID_WORK_GROUP_SIZE",
    "CL_INVALID_WORK_ITEM_SIZE",
    "CL_INVALID_GLOBAL_OFFSET",
    "CL_INVALID_EVENT_WAIT_LIST",
    "CL_INVALID_EVENT",
    "CL_INVALID_OPERATION",
    "CL_INVALID_GL_OBJECT",
    "CL_INVALID_BUFFER_SIZE",
    "CL_INVALID_MIP_LEVEL",
    "CL_INVALID_GLOBAL_WORK_SIZE"
  };

  return cl_error_string[ -1*error];
}

//EOF
