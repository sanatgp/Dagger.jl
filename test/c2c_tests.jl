# test/c2c_tests.jl

using Test
using FFTW
using Random
using Distributed
using Dagger
using CUDA

include("../src/fft.jl")
@testset "C2C FFT Tests" begin
    # 2D Tests
    @testset "2D C2C FFT" begin
        N, M = 64, 32
        input = rand(ComplexF64, N, M)
        
        # Forward transform
        result_fft = fft(input, (FFT(), FFT()), (1, 2))
        expected_fft = FFTW.fft(input, (1, 2))
        @test isapprox(result_fft, expected_fft, rtol=1e-10)
        
        # Backward transform
        result_ifft = ifft(result_fft, (IFFT(), IFFT()), (1, 2))
        @test isapprox(input, result_ifft, rtol=1e-10)
    end

    # 3D Tests
    @testset "3D C2C FFT" begin
        N, M, K = 32, 16, 8
        input = rand(ComplexF64, N, M, K)
        
        # Forward transform
        result_fft = fft(input, (FFT(), FFT(), FFT()), (1, 2, 3))
        expected_fft = FFTW.fft(input, (1, 2, 3))
        @test isapprox(result_fft, expected_fft, rtol=1e-10)
        
        # Backward transform
        result_ifft = ifft(result_fft, (IFFT(), IFFT(), IFFT()), (1, 2, 3))
        @test isapprox(input, result_ifft, rtol=1e-10)
    end
end
