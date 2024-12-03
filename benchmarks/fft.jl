# benchmark/benchmarks.jl

using BenchmarkTools
using FFTW
using Random
using Distributed
using Dagger
using CUDA
include("../src/fft.jl")
const SUITE = BenchmarkGroup()

# Create benchmark groups
SUITE["r2c"] = BenchmarkGroup()
SUITE["c2c"] = BenchmarkGroup()
SUITE["r2r"] = BenchmarkGroup()

# R2C Benchmarks
for dims in [(64, 32), (32, 16, 8)]
    N = length(dims)
    input = rand(Float64, dims...)
    
    SUITE["r2c"][string(N, "D forward")] = @benchmarkable fft($input, $(ntuple(i -> i == 1 ? RFFT() : FFT(), N)), $(ntuple(identity, N)))
    
    fft_result = fft(input, ntuple(i -> i == 1 ? RFFT() : FFT(), N), ntuple(identity, N))
    SUITE["r2c"][string(N, "D backward")] = @benchmarkable ifft($fft_result, $(ntuple(i -> i == 1 ? IRFFT() : IFFT(), N)), $(ntuple(identity, N)), $(dims[1]))
end

# C2C Benchmarks
for dims in [(64, 32), (32, 16, 8)]
    N = length(dims)
    input = rand(ComplexF64, dims...)
    
    SUITE["c2c"][string(N, "D forward")] = @benchmarkable fft($input, $(ntuple(i -> FFT(), N)), $(ntuple(identity, N)))
    
    fft_result = fft(input, ntuple(i -> FFT(), N), ntuple(identity, N))
    SUITE["c2c"][string(N, "D backward")] = @benchmarkable ifft($fft_result, $(ntuple(i -> IFFT(), N)), $(ntuple(identity, N)))
end

# R2R Benchmarks
for dims in [(128,), (64, 32), (32, 16, 8)]
    N = length(dims)
    input = rand(Float64, dims...)
    
    SUITE["r2r"][string(N, "D forward")] = @benchmarkable fft($input, $(ntuple(i -> R2R(FFTW.REDFT10), N)), $(ntuple(identity, N)))
    
    fft_result = fft(input, ntuple(i -> R2R(FFTW.REDFT10), N), ntuple(identity, N))
    SUITE["r2r"][string(N, "D backward")] = @benchmarkable ifft($fft_result, $(ntuple(i -> R2R(FFTW.REDFT01), N)), $(ntuple(identity, N)))
end
