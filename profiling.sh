#!/bin/bash
# profiling.sh
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

rm ./*.log

COMPUTE_PROFILE=1 COMPUTE_PROFILE_CONFIG=nvvp.cfg ./program $1 $2 $3

find . -name 'opencl_profile*.log' | while read FILE;
do
  target=${FILE//opencl/cuda};
  sed -f ./convertToCUDA.sed $FILE > $target;
done

cat cuda_profile_*.log > cuda_profile.log


