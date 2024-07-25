using Test
using Dagger

function arrays_approx_equal(A, B, tol=1e-3)
    return all(abs.(A .- B) .< tol)
end

@testset "DaggerFFT Tests" begin

    # 3D FFT Test
    @testset "3D FFT" begin
        A = rand(ComplexF64, 4, 4, 4)
        transforms = (FFT(), FFT(), FFT())
        dims = (1, 2, 3)
        A_transformed = fft(A, transforms, dims)

        # Transform back to verify
        transforms = (IFFT(), IFFT(), IFFT())
        A_inverse_transformed = ifft(A_transformed, transforms, dims)
        @test arrays_approx_equal(A, A_inverse_transformed)
    end

    # 2D FFT Test
    @testset "2D FFT" begin
        A = rand(ComplexF64, 4, 4)
        transforms = (FFT(), FFT())
        dims = (1, 2)
        A_transformed = fft(A, transforms, dims)

        # Transform back to verify
        transforms = (IFFT(), IFFT())
        A_inverse_transformed = ifft(A_transformed, transforms, dims)
        @test arrays_approx_equal(A, A_inverse_transformed)
    end

    # 1D FFT Test
    @testset "1D FFT" begin
        A = rand(ComplexF64, 4)
        transforms = (FFT(),)
        dims = (1,)
        A_transformed = fft(A, transforms, dims)
        
        # Transform back to verify
        transforms = (IFFT(),)
        A_inverse_transformed = ifft(A_transformed, transforms, dims)
        @test arrays_approx_equal(A, A_inverse_transformed)
    end

    # 3D RFFT Test
    @testset "3D RFFT" begin
        A = rand(Float64, 4, 4, 4)
        transforms = (FFT(), FFT(), FFT())
        dims = (1, 2, 3)
        A_transformed = fft(A, transforms, dims)
        
        # Transform back to verify
        transforms = (IFFT(), IFFT(), IFFT())
        A_inverse_transformed = ifft(A_transformed, transforms, dims)
        @test arrays_approx_equal(A, real(A_inverse_transformed))
    end

    # 2D RFFT Test
    @testset "2D RFFT" begin
        A = rand(Float64, 4, 4)
        transforms = (FFT(), FFT())
        dims = (1, 2)
        A_transformed = fft(A, transforms, dims)
        
        # Transform back to verify
        transforms = (IFFT(), IFFT())
        A_inverse_transformed = ifft(A_transformed, transforms, dims)
        @test arrays_approx_equal(A, real(A_inverse_transformed))
    end

    # 1D RFFT Test
    @testset "1D RFFT" begin
        A = rand(Float64, 4)
        transforms = (FFT(),)
        dims = (1,)
        A_transformed = fft(A, transforms, dims)
        
        # Transform back to verify
        transforms = (IFFT(),)
        A_inverse_transformed = ifft(A_transformed, transforms, dims)
        @test arrays_approx_equal(A, real(A_inverse_transformed))
    end

end
