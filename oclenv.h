/*
# oclenv.h
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

#ifndef  OCLPTX_OCLENV_H_
#define  OCLPTX_OCLENV_H_

#include <iostream>
#include <vector>

#include <CL/cl.hpp>

#include "customtypes.h"

class OclEnv{

  public:

    OclEnv();

    ~OclEnv();

    //
    // Container Set/Get
    //

    cl::Context * GetContext(uint32_t device_num);

    cl::Device * GetDevice(uint32_t device_num);
    uint32_t HowManyDevices();
    uint32_t HowManyCQ();

    cl::CommandQueue * GetCq(uint32_t device_num);
    cl::Kernel * GetKernel(uint32_t kernel_num);

    ConfigData * GetConfigData();

    void SetGPUs(std::vector<uint32_t> gpu_select);

    std::vector<uint32_t> GetGPUs();

    //
    // OpenCL API Interface/Helper Functions
    //

    void OclInit();

    void OclDeviceInfo();

    void NewCLCommandQueues();

    void CreateKernels();

    std::string OclErrorStrings(cl_int error);

    size_t GetKernelWorkGroupInfo(uint32_t device);

    void Die(uint32_t reason, std::string additional = "");

  private:
    //
    // OpenCL Objects
    //
    std::vector<cl::Context> ocl_contexts;

    std::vector<cl::Platform> ocl_platforms;

    std::vector<cl::Device> ocl_devices;

    std::vector<cl::CommandQueue> ocl_device_queues;

    std::vector<cl::Kernel> kernel_set;
    //Every compiled kernel is stored here.

    std::vector<uint32_t> desired_gpus;

    ConfigData config_data;
};

#endif

//EOF
