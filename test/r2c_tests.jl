# test/r2c_tests.jl

using Test
using FFTW
using Random
using Distributed
using Dagger
using CUDA

include("../src/fft.jl")
@testset "R2C FFT Tests" begin
    # 2D Tests
    @testset "2D R2C FFT" begin
        N, M = 64, 64
        input = rand(Float64, N, M)
        
        # Forward transform
        result_fft = fft(input, (RFFT(), FFT()), (1, 2), decomp=Pencil())
        expected_fft = rfft(input, [1, 2])
        @test isapprox(result_fft, expected_fft, rtol=1e-10)
        
        # Backward transform
        result_ifft = ifft(result_fft, (IRFFT(), IFFT()), (1, 2))
        @test isapprox(input, result_ifft, rtol=1e-10)
    end

    # 3D Tests
    @testset "3D R2C FFT" begin
        N, M, K = 16, 16, 16
        input = rand(Float64, N, M, K)
        
        # Forward transform
        result_fft = fft(input, (RFFT(), FFT(), FFT()), (1, 2, 3), decomp=Pencil())
        expected_fft = rfft(input, [1, 2, 3])
        @test isapprox(result_fft, expected_fft, rtol=1e-10)
        
        # Backward transform 
        result_ifft = ifft(result_fft, (IRFFT(), IFFT(), IFFT()), (1, 2, 3))
        @test isapprox(input, result_ifft, rtol=1e-10)
    end
end
