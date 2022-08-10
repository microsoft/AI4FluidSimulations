#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


using Test

@testset begin
    using PyCall, JLD, HDF5
    @test typeof(PyCall) == Module
    @test typeof(JLD) == Module
    @test typeof(HDF5) == Module
    using WaterLily
    @test typeof(WaterLily) == Module
    using AzureClusterlessHPC
    @test typeof(AzureClusterlessHPC) == Module
end