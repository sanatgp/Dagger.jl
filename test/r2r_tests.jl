# test/r2r_tests.jl

using Test
using FFTW
using Random
using Distributed
using Dagger

include("../src/fft.jl")
@testset "R2R FFT Tests" begin

    # 2D Tests
    @testset "2D R2R FFT" begin
        N, M = 64, 64
        input = rand(Float64, N, M)
        
        # Forward transform
        result_dct = fft(input, (R2R(FFTW.REDFT10), R2R(FFTW.REDFT10)), (1, 2))
        expected_dct = FFTW.r2r(input, FFTW.REDFT10, (1, 2))
        @test isapprox(result_dct, expected_dct, rtol=1e-10)
        
        # Backward transform
        result_idct = ifft(result_dct, (R2R(FFTW.REDFT01), R2R(FFTW.REDFT01)), (1, 2))
        @test isapprox(input, result_idct, rtol=1e-10)
    end

    # 3D Tests
    @testset "3D R2R FFT" begin
        N, M, K = 32, 16, 8
        input = rand(Float64, N, M, K)
        
        # Forward transform
        result_dct = fft(input, (R2R(FFTW.REDFT10), R2R(FFTW.REDFT10), R2R(FFTW.REDFT10)), (1, 2, 3))
        expected_dct = FFTW.r2r(input, FFTW.REDFT10, [1, 2, 3])
        @test isapprox(result_dct, expected_dct, rtol=1e-10)
        
        # Backward transform
      #  result_idct = ifft(result_dct, (R2R(FFTW.REDFT01), R2R(FFTW.REDFT01), R2R(FFTW.REDFT01)), (1, 2, 3))
      #  @test isapprox(input, result_idct, rtol=1e-10)
    end
end
